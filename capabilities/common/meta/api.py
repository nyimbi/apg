#!/usr/bin/env python3
"""
APG Metadata Management - REST API
Comprehensive REST API for metadata management operations

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from uuid_extensions import uuid7str

from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError

from .service import (
	APGMetadataService,
	MetaAssetRecord,
	MetaCertificationRecord,
	MetaCatalogAgentRecord,
	MetaClassificationRecord,
	MetaDiscoveryJobRecord,
	MetaGlossaryTermRecord,
	MetaLifecycleBatchRecord,
	MetaLineageRecord,
	MetaQualityRecord,
	MetaService,
	get_metadata_service,
)
from .search_engine import SearchQuery
from .discovery import DiscoverySchedule
from .connectors import ConnectorConfig
from .lineage_engine import LineageEdge


SERVICE = MetaService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"record_count": len(SERVICE.list_records(tenant_id)),
	}


def register_asset_record(**kwargs: Any) -> MetaAssetRecord:
	return SERVICE.register_asset(**kwargs)


def schedule_discovery_record(**kwargs: Any) -> MetaDiscoveryJobRecord:
	return SERVICE.schedule_discovery(**kwargs)


def classify_asset_record(**kwargs: Any) -> MetaClassificationRecord:
	return SERVICE.classify_asset(**kwargs)


def review_classification_record(**kwargs: Any) -> MetaClassificationRecord:
	return SERVICE.review_classification(**kwargs)


def capture_lineage_record(**kwargs: Any) -> MetaLineageRecord:
	return SERVICE.capture_lineage(**kwargs)


def assess_quality_record(**kwargs: Any) -> MetaQualityRecord:
	return SERVICE.assess_quality(**kwargs)


def request_certification_record(**kwargs: Any) -> MetaCertificationRecord:
	return SERVICE.request_certification(**kwargs)


def register_glossary_term_record(**kwargs: Any) -> MetaGlossaryTermRecord:
	return SERVICE.register_glossary_term(**kwargs)


def publish_asset_record(**kwargs: Any) -> MetaAssetRecord:
	return SERVICE.publish_asset(**kwargs)


def register_catalog_agent(**kwargs: Any) -> MetaCatalogAgentRecord:
	return SERVICE.register_catalog_agent(**kwargs)


def validate_meta_lifecycle_batch(**kwargs: Any) -> MetaLifecycleBatchRecord:
	return SERVICE.validate_meta_lifecycle_batch(**kwargs)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id, record_type)


def list_metadata(tenant_id: str | None = None) -> dict[str, Any]:
	return {
		"summary": SERVICE.dashboard_summary(tenant_id),
		"assets": SERVICE.list_records(tenant_id, "assets"),
		"discovery_jobs": SERVICE.list_records(tenant_id, "discovery_jobs"),
		"classifications": SERVICE.list_records(tenant_id, "classifications"),
		"lineage": SERVICE.list_records(tenant_id, "lineage"),
		"quality_assessments": SERVICE.list_records(tenant_id, "quality_assessments"),
		"certifications": SERVICE.list_records(tenant_id, "certifications"),
		"glossary_terms": SERVICE.list_records(tenant_id, "glossary_terms"),
		"catalog_agents": SERVICE.list_records(tenant_id, "catalog_agents"),
		"lifecycle_batches": SERVICE.list_records(tenant_id, "lifecycle_batches"),
		"audit_events": SERVICE.list_records(tenant_id, "audit_events"),
	}


# Create API namespace
meta_api = Namespace('metadata', description='Metadata Management Operations')


# === Request/Response Models ===

# Discovery models
discovery_schedule_model = meta_api.model('DiscoverySchedule', {
	'name': fields.String(required=True, description='Schedule name'),
	'description': fields.String(description='Schedule description'),
	'connector_type': fields.String(required=True, description='Connector type'),
	'connection_params': fields.Raw(required=True, description='Connection parameters'),
	'schedule_cron': fields.String(description='Cron schedule expression'),
	'is_enabled': fields.Boolean(default=True, description='Whether schedule is enabled'),
	'is_one_time': fields.Boolean(default=False, description='One-time discovery')
})

discovery_job_response = meta_api.model('DiscoveryJobResponse', {
	'job_id': fields.String(description='Discovery job ID'),
	'status': fields.String(description='Job status'),
	'message': fields.String(description='Status message')
})

# Search models
search_request_model = meta_api.model('SearchRequest', {
	'query_text': fields.String(required=True, description='Search query text'),
	'filters': fields.Raw(description='Additional search filters'),
	'limit': fields.Integer(default=50, description='Maximum results'),
	'offset': fields.Integer(default=0, description='Result offset'),
	'enable_natural_language': fields.Boolean(default=True, description='Enable NLP search'),
	'search_types': fields.List(fields.String, description='Asset types to search')
})

search_response_model = meta_api.model('SearchResponse', {
	'results': fields.List(fields.Raw, description='Search results'),
	'total_results': fields.Integer(description='Total matching results'),
	'query_time_ms': fields.Float(description='Query execution time'),
	'suggestions': fields.List(fields.String, description='Query suggestions')
})

# Asset models
asset_response_model = meta_api.model('AssetResponse', {
	'id': fields.String(description='Asset ID'),
	'name': fields.String(description='Asset name'),
	'display_name': fields.String(description='Display name'),
	'description': fields.String(description='Asset description'),
	'asset_type': fields.String(description='Asset type'),
	'source_system': fields.String(description='Source system'),
	'status': fields.String(description='Asset status'),
	'quality_score': fields.Float(description='Data quality score'),
	'tags': fields.List(fields.String, description='Asset tags'),
	'created_at': fields.DateTime(description='Creation timestamp'),
	'updated_at': fields.DateTime(description='Last update timestamp')
})

# Lineage models
lineage_request_model = meta_api.model('LineageRequest', {
	'direction': fields.String(default='both', description='Lineage direction (upstream/downstream/both)'),
	'max_depth': fields.Integer(default=5, description='Maximum depth to traverse'),
	'include_columns': fields.Boolean(default=False, description='Include column-level lineage')
})

lineage_edge_model = meta_api.model('LineageEdge', {
	'source_asset_id': fields.String(required=True, description='Source asset ID'),
	'target_asset_id': fields.String(required=True, description='Target asset ID'),
	'lineage_type': fields.String(required=True, description='Lineage type'),
	'transformation_logic': fields.String(description='Transformation description'),
	'confidence_score': fields.Float(description='Confidence score')
})

# Classification models
classification_request_model = meta_api.model('ClassificationRequest', {
	'column_name': fields.String(required=True, description='Column name'),
	'data_type': fields.String(required=True, description='Data type'),
	'sample_data': fields.List(fields.Raw, required=True, description='Sample data'),
	'context': fields.Raw(description='Additional context')
})

# Health models
health_response_model = meta_api.model('HealthResponse', {
	'service_name': fields.String(description='Service name'),
	'status': fields.String(description='Service status'),
	'uptime_seconds': fields.Float(description='Service uptime'),
	'components': fields.Raw(description='Component health status'),
	'metrics': fields.Raw(description='Service metrics'),
	'issues': fields.Raw(description='Service issues')
})


# === API Resources ===

@meta_api.route('/health')
class HealthResource(Resource):
	@meta_api.marshal_with(health_response_model)
	async def get(self):
		"""Get service health status"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'status': 'error', 'message': 'Service not initialized'}, 503
			
			health_status = await service.get_health_status()
			return health_status, 200
			
		except Exception as e:
			current_app.logger.error(f"Health check failed: {str(e)}")
			return {'status': 'error', 'message': str(e)}, 500


@meta_api.route('/metrics')
class MetricsResource(Resource):
	async def get(self):
		"""Get service performance metrics"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			metrics = await service.get_service_metrics()
			return metrics, 200
			
		except Exception as e:
			current_app.logger.error(f"Metrics retrieval failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/discovery/schedules')
class DiscoverySchedulesResource(Resource):
	@meta_api.expect(discovery_schedule_model)
	@meta_api.marshal_with(discovery_job_response)
	async def post(self):
		"""Create a new discovery schedule"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			data = request.get_json()
			tenant_id = request.headers.get('X-Tenant-ID', 'default')
			
			# Create connector config
			connector_config = ConnectorConfig(
				name=data.get('name'),
				connector_type=data.get('connector_type'),
				connection_params=data.get('connection_params'),
				tenant_id=tenant_id
			)
			
			# Create discovery schedule
			schedule = DiscoverySchedule(
				name=data.get('name'),
				description=data.get('description'),
				connector_config=connector_config,
				schedule_cron=data.get('schedule_cron'),
				tenant_id=tenant_id,
				is_enabled=data.get('is_enabled', True),
				is_one_time=data.get('is_one_time', False)
			)
			
			schedule_id = await service.create_discovery_schedule(schedule)
			
			return {
				'job_id': schedule_id,
				'status': 'created',
				'message': 'Discovery schedule created successfully'
			}, 201
			
		except Exception as e:
			current_app.logger.error(f"Discovery schedule creation failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/discovery/jobs/<string:schedule_id>/run')
class DiscoveryJobRunResource(Resource):
	@meta_api.marshal_with(discovery_job_response)
	async def post(self, schedule_id: str):
		"""Run a discovery job"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			data = request.get_json() or {}
			override_config = data.get('override_config', {})
			
			job_id = await service.run_discovery(schedule_id, override_config)
			
			return {
				'job_id': job_id,
				'status': 'started',
				'message': 'Discovery job started successfully'
			}, 200
			
		except Exception as e:
			current_app.logger.error(f"Discovery job execution failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/discovery/jobs/<string:job_id>')
class DiscoveryJobResource(Resource):
	async def get(self, job_id: str):
		"""Get discovery job status"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			job_status = await service.get_discovery_job_status(job_id)
			
			if not job_status:
				return {'error': 'Job not found'}, 404
			
			return job_status, 200
			
		except Exception as e:
			current_app.logger.error(f"Discovery job status retrieval failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/search')
class SearchResource(Resource):
	@meta_api.expect(search_request_model)
	@meta_api.marshal_with(search_response_model)
	async def post(self):
		"""Search metadata assets"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			data = request.get_json()
			tenant_id = request.headers.get('X-Tenant-ID', 'default')
			
			# Create search query
			search_query = SearchQuery(
				query_text=data.get('query_text'),
				tenant_id=tenant_id,
				filters=data.get('filters', {}),
				limit=data.get('limit', 50),
				offset=data.get('offset', 0),
				enable_natural_language=data.get('enable_natural_language', True),
				search_types=data.get('search_types', [])
			)
			
			results = await service.search_metadata(search_query)
			return results, 200
			
		except Exception as e:
			current_app.logger.error(f"Search failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/assets')
class AssetsResource(Resource):
	async def get(self):
		"""List metadata assets"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			tenant_id = request.headers.get('X-Tenant-ID', 'default')
			
			# Get query parameters
			limit = int(request.args.get('limit', 100))
			offset = int(request.args.get('offset', 0))
			
			# Parse filters from query parameters
			filters = {}
			for key, value in request.args.items():
				if key not in ['limit', 'offset']:
					if ',' in value:
						filters[key] = value.split(',')
					else:
						filters[key] = value
			
			assets = await service.list_assets(tenant_id, filters, limit, offset)
			return assets, 200
			
		except Exception as e:
			current_app.logger.error(f"Asset listing failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/assets/<string:asset_id>')
class AssetResource(Resource):
	@meta_api.marshal_with(asset_response_model)
	async def get(self, asset_id: str):
		"""Get metadata asset by ID"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			tenant_id = request.headers.get('X-Tenant-ID', 'default')
			
			asset = await service.get_asset(asset_id, tenant_id)
			
			if not asset:
				return {'error': 'Asset not found'}, 404
			
			return asset, 200
			
		except Exception as e:
			current_app.logger.error(f"Asset retrieval failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/assets/<string:asset_id>/lineage')
class AssetLineageResource(Resource):
	@meta_api.expect(lineage_request_model, validate=False)
	async def get(self, asset_id: str):
		"""Get asset lineage"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			tenant_id = request.headers.get('X-Tenant-ID', 'default')
			
			# Get query parameters
			direction = request.args.get('direction', 'both')
			max_depth = int(request.args.get('max_depth', 5))
			
			lineage_paths = await service.get_lineage_path(
				asset_id, tenant_id, direction, max_depth
			)
			
			return {
				'asset_id': asset_id,
				'lineage_paths': lineage_paths,
				'direction': direction,
				'max_depth': max_depth
			}, 200
			
		except Exception as e:
			current_app.logger.error(f"Lineage retrieval failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/lineage')
class LineageResource(Resource):
	@meta_api.expect(lineage_edge_model)
	async def post(self):
		"""Add lineage relationship"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			data = request.get_json()
			tenant_id = request.headers.get('X-Tenant-ID', 'default')
			
			# Create lineage edge
			edge = LineageEdge(
				source_asset_id=data.get('source_asset_id'),
				target_asset_id=data.get('target_asset_id'),
				lineage_type=data.get('lineage_type'),
				transformation_logic=data.get('transformation_logic'),
				confidence_score=data.get('confidence_score', 1.0),
				tenant_id=tenant_id
			)
			
			edge_id = await service.add_lineage_relationship(edge)
			
			return {
				'edge_id': edge_id,
				'status': 'created',
				'message': 'Lineage relationship added successfully'
			}, 201
			
		except Exception as e:
			current_app.logger.error(f"Lineage creation failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/assets/<string:asset_id>/impact')
class AssetImpactResource(Resource):
	async def post(self, asset_id: str):
		"""Analyze asset impact"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			data = request.get_json() or {}
			tenant_id = request.headers.get('X-Tenant-ID', 'default')
			
			change_type = data.get('change_type', 'schema_change')
			change_details = data.get('change_details', {})
			
			impact_analysis = await service.analyze_impact(
				asset_id, tenant_id, change_type, change_details
			)
			
			return impact_analysis, 200
			
		except Exception as e:
			current_app.logger.error(f"Impact analysis failed: {str(e)}")
			return {'error': str(e)}, 500


@meta_api.route('/classification/classify')
class ClassificationResource(Resource):
	@meta_api.expect(classification_request_model)
	async def post(self):
		"""Classify column data using AI"""
		try:
			service = await get_metadata_service()
			if not service:
				return {'error': 'Service not initialized'}, 503
			
			data = request.get_json()
			
			result = await service.classify_column_data(
				column_name=data.get('column_name'),
				data_type=data.get('data_type'),
				sample_data=data.get('sample_data'),
				context=data.get('context', {})
			)
			
			return result, 200
			
		except Exception as e:
			current_app.logger.error(f"Classification failed: {str(e)}")
			return {'error': str(e)}, 500


# === Utility Functions ===

def create_metadata_api() -> Api:
	"""Create the metadata management API"""
	api = Api(
		version='1.0',
		title='APG Metadata Management API',
		description='Tenant-scoped metadata catalog, discovery, lineage, and governance API',
		doc='/docs/',
		prefix='/api/v1'
	)
	
	api.add_namespace(meta_api, path='/metadata')
	return api


def register_api_routes(blueprint: Blueprint) -> None:
	"""Register API routes with a Flask blueprint"""
	api = create_metadata_api()
	api.init_app(blueprint)


# === Error Handlers ===

def handle_async_view(f):
	"""Decorator to handle async views in Flask"""
	def wrapper(*args, **kwargs):
		if asyncio.iscoroutinefunction(f):
			# Create new event loop for each request if needed
			try:
				loop = asyncio.get_event_loop()
			except RuntimeError:
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
			
			return loop.run_until_complete(f(*args, **kwargs))
		else:
			return f(*args, **kwargs)
	return wrapper


# Apply async handler to all resources
for resource_class in [
	HealthResource, MetricsResource, DiscoverySchedulesResource,
	DiscoveryJobRunResource, DiscoveryJobResource, SearchResource,
	AssetsResource, AssetResource, AssetLineageResource,
	LineageResource, AssetImpactResource, ClassificationResource
]:
	for method in ['get', 'post', 'put', 'delete', 'patch']:
		if hasattr(resource_class, method):
			original_method = getattr(resource_class, method)
			if asyncio.iscoroutinefunction(original_method):
				setattr(resource_class, method, handle_async_view(original_method))


# === API Documentation ===

@meta_api.doc('metadata_api_info')
@meta_api.route('/info')
class APIInfoResource(Resource):
	def get(self):
		"""Get API information and capabilities"""
		return {
			'api_name': 'APG Metadata Management API',
			'version': '1.0.0',
			'description': 'Tenant-scoped metadata catalog, discovery, lineage, and governance API',
			'capabilities': [
				'auto_discovery',
				'ai_classification',
				'lineage_tracking',
				'natural_language_search',
				'impact_analysis',
				'real_time_monitoring'
			],
			'endpoints': {
				'health': 'GET /metadata/health - Service health status',
				'metrics': 'GET /metadata/metrics - Performance metrics',
				'discovery': {
					'schedules': 'POST /metadata/discovery/schedules - Create discovery schedule',
					'run': 'POST /metadata/discovery/jobs/{schedule_id}/run - Run discovery job',
					'status': 'GET /metadata/discovery/jobs/{job_id} - Get job status'
				},
				'search': 'POST /metadata/search - Search metadata assets',
				'assets': {
					'list': 'GET /metadata/assets - List assets',
					'get': 'GET /metadata/assets/{asset_id} - Get asset details',
					'lineage': 'GET /metadata/assets/{asset_id}/lineage - Get asset lineage',
					'impact': 'POST /metadata/assets/{asset_id}/impact - Analyze impact'
				},
				'lineage': 'POST /metadata/lineage - Add lineage relationship',
				'classification': 'POST /metadata/classification/classify - Classify data'
			},
			'authentication': 'X-Tenant-ID header required for multi-tenant operations',
			'documentation': '/api/v1/docs/ - Interactive API documentation'
		}, 200

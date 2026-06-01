#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) REST API - Flask-AppBuilder Implementation
Real RESTful API endpoints using Flask-AppBuilder for the DVRL capability

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid_extensions import uuid7str

# Real Flask-AppBuilder imports
from flask import Flask, Blueprint, request, jsonify, g, current_app
from flask_appbuilder import AppBuilder, BaseView, ModelView, expose, has_access
from flask_appbuilder.api import BaseApi
try:
	from flask_appbuilder.api import expose_api
except ImportError:  # pragma: no cover - Flask-AppBuilder version compatibility
	from flask_appbuilder.api import expose as expose_api
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.models.sqla.interface import SQLAInterface
try:
	from flask_appbuilder.security import current_user
except ImportError:  # pragma: no cover - Flask-AppBuilder version compatibility
	from flask_login import current_user
from flask_appbuilder.baseviews import BaseModelView
from werkzeug.exceptions import BadRequest, Unauthorized, NotFound, InternalServerError
import json

# DVRL Flask-AppBuilder Blueprint
dvrl_blueprint = Blueprint(
	'dvrl_api',
	__name__,
	url_prefix='/api/v1/dvrl'
)

# DVRL API Controller using Flask-AppBuilder
class DVRLAPIController(BaseApi):
	"""Real REST API controller using Flask-AppBuilder for DVRL operations"""
	
	resource_name = 'dvrl'
	allow_browser_login = True
	
	def __init__(self, dvrl_service):
		super().__init__()
		self.dvrl_service = dvrl_service
		self.api_version = "v1"
		
	def _execute_async(self, coro):
		"""Execute async operation in event loop"""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			return loop.run_until_complete(coro)
		finally:
			loop.close()
	
	# Data Source Management Endpoints
	@expose_api('/data-sources', methods=['POST'])
	@protect()
	def register_data_source(self):
		"""POST /api/v1/dvrl/data-sources - Register new data source"""
		try:
			if not request.is_json:
				raise BadRequest("Content-Type must be application/json")
			
			source_config = request.get_json()
			
			# Validate required fields
			if not source_config.get('name') or not source_config.get('type'):
				raise BadRequest('Missing required fields: name, type')
			
			# Check authorization using Flask-AppBuilder security
			if not self.appbuilder.sm.has_access('can_create', 'DataSourceModelView'):
				raise Unauthorized()
			
			# Execute async operation
			data_source = self._execute_async(
				self.dvrl_service.register_data_source(source_config)
			)
			
			response_data = {
				'data_source_id': data_source.id,
				'name': data_source.name,
				'type': data_source.type.value,
				'status': data_source.status.value,
				'created_at': data_source.created_at.isoformat(),
				'created_by': current_user.username if current_user else "system"
			}
			
			return jsonify(response_data), 201
			
		except BadRequest as e:
			return jsonify({'error': str(e)}), 400
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Data source registration failed: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	@expose_api('/data-sources', methods=['GET'])
	@protect()
	def get_data_sources(self):
		"""GET /api/v1/dvrl/data-sources - List data sources"""
		try:
			# Check authorization
			if not self.appbuilder.sm.has_access('can_list', 'DataSourceModelView'):
				raise Unauthorized()
			
			data_sources = []
			for source in self.dvrl_service.data_sources.values():
				data_sources.append({
					'id': source.id,
					'name': source.name,
					'type': source.type.value,
					'status': source.status.value,
					'created_at': source.created_at.isoformat(),
					'query_count': source.query_count
				})
			
			return jsonify({
				'data_sources': data_sources,
				'total_count': len(data_sources)
			})
			
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Failed to get data sources: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	@expose_api('/data-sources/<source_id>/schema', methods=['GET'])
	@protect()
	def get_data_source_schema(self, source_id: str):
		"""GET /api/v1/dvrl/data-sources/{id}/schema - Get data source schema"""
		try:
			# Check authorization
			if not self.appbuilder.sm.has_access('can_show', 'DataSourceModelView'):
				raise Unauthorized()
			
			if source_id not in self.dvrl_service.data_sources:
				raise NotFound('Data source not found')
			
			schema = self._execute_async(
				self.dvrl_service.get_data_source_schema(source_id)
			)
			
			return jsonify({
				'schema_id': schema.id,
				'data_source_id': source_id,
				'schema_name': schema.schema_name,
				'tables': schema.tables,
				'discovery_method': schema.discovery_method,
				'confidence_score': schema.confidence_score,
				'created_at': schema.created_at.isoformat()
			})
			
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except NotFound as e:
			return jsonify({'error': 'Data source not found'}), 404
		except Exception as e:
			current_app.logger.error(f"Failed to get schema for {source_id}: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	# Query Execution Endpoints
	@expose_api('/query/execute', methods=['POST'])
	@protect()
	def execute_sql_query(self):
		"""POST /api/v1/dvrl/query/execute - Execute federated SQL query"""
		try:
			if not request.is_json:
				raise BadRequest("Content-Type must be application/json")
			
			query_request = request.get_json()
			
			# Validate required fields
			if not query_request.get('sql'):
				raise BadRequest('Missing required field: sql')
			
			# Check authorization
			if not self.appbuilder.sm.has_access('can_query', 'DVRLQueryView'):
				raise Unauthorized()
			
			# Execute federated query
			query_result = self._execute_async(
				self.dvrl_service.execute_federated_query(
					query_request['sql'],
					query_request.get('parameters', {}),
					query_request.get('options', {})
				)
			)
			
			response_data = {
				'query_id': query_result.id,
				'sql': query_result.original_sql,
				'status': query_result.status.value,
				'results': query_result.results if hasattr(query_result, 'results') else [],
				'rows_returned': query_result.rows_returned,
				'bytes_processed': query_result.bytes_processed,
				'duration_ms': query_result.duration_ms,
				'cache_used': query_result.cache_used,
				'executed_at': query_result.created_at.isoformat(),
				'executed_by': current_user.username if current_user else "system"
			}
			
			return jsonify(response_data)
			
		except BadRequest as e:
			return jsonify({'error': str(e)}), 400
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Query execution failed: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	@expose_api('/query/natural-language', methods=['POST'])
	@protect()
	def execute_natural_language_query(self):
		"""POST /api/v1/dvrl/query/natural-language - Execute natural language query"""
		try:
			if not request.is_json:
				raise BadRequest("Content-Type must be application/json")
			
			nl_request = request.get_json()
			
			# Validate required fields
			if not nl_request.get('query'):
				raise BadRequest('Missing required field: query')
			
			# Check authorization
			if not self.appbuilder.sm.has_access('can_query', 'DVRLQueryView'):
				raise Unauthorized()
			
			# Execute natural language query
			result = self._execute_async(
				self.dvrl_service.execute_natural_language_query(
					nl_request['query'],
					nl_request.get('data_sources', []),
					nl_request.get('options', {})
				)
			)
			
			response_data = {
				'query_id': result.get('query_id'),
				'natural_language_query': nl_request['query'],
				'generated_sql': result.get('sql'),
				'confidence': result.get('confidence'),
				'results': result.get('results', []),
				'execution_time_ms': result.get('execution_time_ms'),
				'executed_by': current_user.username if current_user else "system"
			}
			
			return jsonify(response_data)
			
		except BadRequest as e:
			return jsonify({'error': str(e)}), 400
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Natural language query failed: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	@expose_api('/query/suggestions', methods=['GET'])
	@protect()
	def get_query_suggestions(self):
		"""GET /api/v1/dvrl/query/suggestions - Get query suggestions"""
		try:
			# Check authorization
			if not self.appbuilder.sm.has_access('can_query', 'DVRLQueryView'):
				raise Unauthorized()
			
			context = request.args.get('context', '')
			suggestions = self._execute_async(
				self.dvrl_service.get_query_suggestions(context)
			)
			
			return jsonify({
				'suggestions': suggestions,
				'context': context,
				'generated_at': datetime.now(timezone.utc).isoformat()
			})
			
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Query suggestions failed: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	# Streaming Query Endpoints
	@expose_api('/streaming/start', methods=['POST'])
	@protect()
	def start_streaming_query(self):
		"""POST /api/v1/dvrl/streaming/start - Start streaming query"""
		try:
			if not request.is_json:
				raise BadRequest("Content-Type must be application/json")
			
			stream_request = request.get_json()
			
			# Validate required fields
			if not stream_request.get('sql'):
				raise BadRequest('Missing required field: sql')
			
			# Check authorization
			if not self.appbuilder.sm.has_access('can_stream', 'DVRLStreamView'):
				raise Unauthorized()
			
			# Start streaming query
			stream_id = self._execute_async(
				self.dvrl_service.execute_streaming_query(
					stream_request['sql'],
					stream_request.get('options', {})
				)
			)
			
			return jsonify({
				'stream_id': stream_id,
				'status': 'started',
				'started_by': current_user.username if current_user else "system",
				'started_at': datetime.now(timezone.utc).isoformat()
			})
			
		except BadRequest as e:
			return jsonify({'error': str(e)}), 400
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Streaming query start failed: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	@expose_api('/streaming/<stream_id>/stop', methods=['POST'])
	@protect()
	def stop_streaming_query(self, stream_id: str):
		"""POST /api/v1/dvrl/streaming/{stream_id}/stop - Stop streaming query"""
		try:
			# Check authorization
			if not self.appbuilder.sm.has_access('can_stream', 'DVRLStreamView'):
				raise Unauthorized()
			
			result = self._execute_async(
				self.dvrl_service.streaming_executor.stop_streaming_query(stream_id)
			)
			
			return jsonify(result)
			
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Streaming query stop failed: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	# Virtual Table Management
	@expose_api('/virtual-tables', methods=['POST'])
	@protect()
	def create_virtual_table(self):
		"""POST /api/v1/dvrl/virtual-tables - Create virtual table"""
		try:
			if not request.is_json:
				raise BadRequest("Content-Type must be application/json")
			
			table_config = request.get_json()
			
			# Validate required fields
			if not table_config.get('name'):
				raise BadRequest('Missing required field: name')
			
			# Check authorization
			if not self.appbuilder.sm.has_access('can_create', 'VirtualTableModelView'):
				raise Unauthorized()
			
			virtual_table = self._execute_async(
				self.dvrl_service.create_virtual_table(table_config)
			)
			
			response_data = {
				'virtual_table_id': virtual_table.id,
				'name': virtual_table.name,
				'definition': virtual_table.definition,
				'created_at': virtual_table.created_at.isoformat(),
				'created_by': current_user.username if current_user else "system"
			}
			
			return jsonify(response_data), 201
			
		except BadRequest as e:
			return jsonify({'error': str(e)}), 400
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Virtual table creation failed: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	# Health and Monitoring Endpoints
	@expose_api('/health', methods=['GET'])
	def get_health_status(self):
		"""GET /api/v1/dvrl/health - Get system health status"""
		try:
			health_status = self._execute_async(
				self.dvrl_service.get_health_status()
			)
			
			return jsonify(health_status)
			
		except Exception as e:
			current_app.logger.error(f"Health check failed: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	@expose_api('/metrics', methods=['GET'])
	@protect()
	def get_performance_metrics(self):
		"""GET /api/v1/dvrl/metrics - Get performance metrics"""
		try:
			# Check authorization
			if not self.appbuilder.sm.has_access('can_metrics', 'DVRLAdminView'):
				raise Unauthorized()
			
			metrics = self._execute_async(
				self.dvrl_service.get_performance_metrics()
			)
			
			return jsonify(metrics)
			
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Metrics retrieval failed: {str(e)}")
			return jsonify({'error': str(e)}), 500
	
	@expose_api('/connectors/stats', methods=['GET'])
	@protect()
	def get_connector_stats(self):
		"""GET /api/v1/dvrl/connectors/stats - Get connector statistics"""
		try:
			# Check authorization
			if not self.appbuilder.sm.has_access('can_admin', 'DVRLAdminView'):
				raise Unauthorized()
			
			connector_stats = self._execute_async(
				self.dvrl_service.connector_manager.get_connector_stats()
			)
			
			return jsonify(connector_stats)
			
		except Unauthorized as e:
			return jsonify({'error': 'Unauthorized'}), 403
		except Exception as e:
			current_app.logger.error(f"Connector stats retrieval failed: {str(e)}")
			return jsonify({'error': str(e)}), 500

# Flask-AppBuilder Model Views for DVRL entities
class DataSourceModelView(ModelView):
	"""Data Source management view"""
	datamodel = None
	list_columns = ['name', 'type', 'status', 'created_at']
	show_columns = ['name', 'type', 'status', 'connection_config', 'created_at', 'updated_at']
	add_columns = ['name', 'type', 'connection_config', 'description']
	edit_columns = ['name', 'connection_config', 'description']

class DVRLQueryView(BaseView):
	"""DVRL Query execution view"""
	default_view = 'execute'
	
	@expose('/execute')
	@has_access
	def execute(self):
		return self.render_template('dvrl/query_execute.html')

class DVRLStreamView(BaseView):
	"""DVRL Streaming query view"""
	default_view = 'manage'
	
	@expose('/manage')
	@has_access
	def manage(self):
		return self.render_template('dvrl/stream_manage.html')

class VirtualTableModelView(ModelView):
	"""Virtual Table management view"""
	datamodel = None
	list_columns = ['name', 'definition', 'created_at']

class DVRLAdminView(BaseView):
	"""DVRL Administration view"""
	default_view = 'dashboard'
	
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		return self.render_template('dvrl/admin_dashboard.html')

# Factory function to create and configure DVRL API
def create_dvrl_api(app: Flask, dvrl_service) -> DVRLAPIController:
	"""Create and configure DVRL API with Flask-AppBuilder"""
	
	appbuilder = AppBuilder(app, session=None)
	
	# Register API controller
	api_controller = DVRLAPIController(dvrl_service)
	appbuilder.add_api(api_controller)
	
	# Register Model Views
	appbuilder.add_view(DataSourceModelView, "Data Sources", category="DVRL")
	appbuilder.add_view(DVRLQueryView, "Query Interface", category="DVRL") 
	appbuilder.add_view(DVRLStreamView, "Streaming Queries", category="DVRL")
	appbuilder.add_view(VirtualTableModelView, "Virtual Tables", category="DVRL")
	appbuilder.add_view(DVRLAdminView, "Administration", category="DVRL")
	
	return api_controller

from .service import DVRLLifecycleService


SERVICE = DVRLLifecycleService()


def capability_status(tenant_id: str = "default") -> Dict[str, Any]:
	"""Return generated-application DVRL capability status."""
	return {
		"capability": "dvrl",
		"tenant_id": tenant_id,
		"summary": SERVICE.dashboard_summary(tenant_id),
		"contract": SERVICE.describe(tenant_id),
	}


def register_source_record(**kwargs) -> Dict[str, Any]:
	"""Register a virtual source through the dependency-light lifecycle service."""
	return SERVICE.register_source(**kwargs).__dict__


def activate_source_record(**kwargs) -> Dict[str, Any]:
	"""Evaluate source activation guardrails."""
	return SERVICE.activate_source(**kwargs).__dict__


def refresh_schema_record(**kwargs) -> Dict[str, Any]:
	"""Refresh source schema metadata through generated-app guardrails."""
	return SERVICE.refresh_schema(**kwargs).__dict__


def publish_virtual_table_record(**kwargs) -> Dict[str, Any]:
	"""Publish a governed virtual table record."""
	return SERVICE.publish_virtual_table(**kwargs).__dict__


def execute_query_record(**kwargs) -> Dict[str, Any]:
	"""Evaluate and record a governed federated query request."""
	return SERVICE.execute_query(**kwargs).__dict__


def cache_result_record(**kwargs) -> Dict[str, Any]:
	"""Evaluate and record a query-cache lifecycle request."""
	return SERVICE.cache_result(**kwargs).__dict__


def change_policy_record(**kwargs) -> Dict[str, Any]:
	"""Evaluate and record a virtualization policy change."""
	return SERVICE.change_policy(**kwargs).__dict__


def retire_source_record(**kwargs) -> Dict[str, Any]:
	"""Evaluate source retirement guardrails."""
	return SERVICE.retire_source(**kwargs).__dict__


def register_virtualization_agent_record(**kwargs) -> Dict[str, Any]:
	"""Register a governed virtualization agent for generated-app composition."""
	return SERVICE.register_virtualization_agent(**kwargs).__dict__


def validate_dvrl_lifecycle_batch_record(**kwargs) -> Dict[str, Any]:
	"""Validate that a DVRL lifecycle batch is routed through Bytewax."""
	return SERVICE.validate_dvrl_lifecycle_batch(**kwargs).__dict__


def list_pending_reviews(tenant_id: str = "default") -> List[Dict[str, Any]]:
	"""List DVRL records awaiting generated-app or operator review."""
	return SERVICE.list_pending_reviews(tenant_id)


def list_records(tenant_id: str = "default", record_type: str | None = None) -> List[Dict[str, Any]]:
	"""List dependency-light DVRL lifecycle records."""
	return SERVICE.list_records(tenant_id, record_type)


def list_metadata(tenant_id: str = "default") -> Dict[str, Any]:
	"""Return DVRL metadata for generated application composition."""
	return {
		"status": capability_status(tenant_id),
		"sources": SERVICE.list_records(tenant_id, "sources"),
		"schemas": SERVICE.list_records(tenant_id, "schemas"),
		"virtual_tables": SERVICE.list_records(tenant_id, "virtual_tables"),
		"queries": SERVICE.list_records(tenant_id, "queries"),
		"caches": SERVICE.list_records(tenant_id, "caches"),
		"policies": SERVICE.list_records(tenant_id, "policies"),
		"virtualization_agents": SERVICE.list_records(tenant_id, "virtualization_agents"),
		"lifecycle_batches": SERVICE.list_records(tenant_id, "lifecycle_batches"),
		"pending_reviews": SERVICE.list_pending_reviews(tenant_id),
		"audit_events": SERVICE.list_records(tenant_id, "audit_events"),
	}


__all__ = [
	"DVRLAPIController",
	"DataSourceModelView",
	"DVRLQueryView", 
	"DVRLStreamView",
	"VirtualTableModelView",
	"DVRLAdminView",
	"create_dvrl_api",
	"dvrl_blueprint",
	"capability_status",
	"register_source_record",
	"activate_source_record",
	"refresh_schema_record",
	"publish_virtual_table_record",
	"execute_query_record",
	"cache_result_record",
	"change_policy_record",
	"retire_source_record",
	"register_virtualization_agent_record",
	"validate_dvrl_lifecycle_batch_record",
	"list_pending_reviews",
	"list_records",
	"list_metadata",
]

#!/usr/bin/env python3
"""
Registry (regy) - APG REST API Implementation
============================================

Comprehensive REST API endpoints for service registry with intelligent discovery,
health monitoring, and seamless APG ecosystem integration.

Author: APG Platform Team
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from urllib.parse import unquote

from flask import Blueprint, request, jsonify, g
try:
	from flask_restx import Api, Resource, Namespace, fields
except ImportError:
	import inspect
	from asyncio import events

	class Resource:
		"""Minimal Flask-RESTX Resource fallback."""
		pass

	class _FieldFactory:
		def __getattr__(self, name):
			def create(*args, **kwargs):
				return {"type": name, "args": args, "kwargs": kwargs}
			return create

	fields = _FieldFactory()

	class Namespace:
		"""Small namespace that registers Resource classes on the API blueprint."""

		def __init__(self, name: str, description: str = ""):
			self.name = name
			self.description = description
			self._api = None
			self._path = f"/{name}"

		def attach(self, api: "Api", path: str | None = None) -> None:
			self._api = api
			self._path = path or f"/{self.name}"

		def model(self, name: str, model: Dict[str, Any]) -> Dict[str, Any]:
			return model

		def route(self, rule: str):
			def decorator(resource_cls):
				if self._api:
					self._api._register_resource(resource_cls, f"{self._path}{rule}")
				return resource_cls
			return decorator

		def param(self, *args, **kwargs):
			return lambda resource_cls: resource_cls

		def doc(self, *args, **kwargs):
			return lambda func: func

		def errorhandler(self, exception_type):
			def decorator(handler):
				return handler
			return decorator

		def marshal_with(self, *args, **kwargs):
			return lambda func: func

		def marshal_list_with(self, *args, **kwargs):
			return lambda func: func

		def expect(self, *args, **kwargs):
			return lambda func: func

	class Api:
		"""Small Flask-RESTX Api fallback for local tests."""

		def __init__(self, blueprint: Blueprint, version: str = "", title: str = "",
					 description: str = "", doc: str | None = None, prefix: str = ""):
			self.blueprint = blueprint
			self.version = version
			self.title = title
			self.description = description
			self.doc_path = doc
			self.prefix = prefix.rstrip("/")
			if doc:
				blueprint.add_url_rule(doc, f"{blueprint.name}_docs", self._docs)

		def _docs(self):
			return "<html><body>Registry API documentation</body></html>"

		def add_namespace(self, namespace: Namespace, path: str | None = None) -> None:
			namespace.attach(self, path)

		def route(self, rule: str):
			def decorator(resource_cls):
				self._register_resource(resource_cls, rule)
				return resource_cls
			return decorator

		def doc(self, *args, **kwargs):
			return lambda func: func

		def errorhandler(self, exception_type):
			def decorator(handler):
				return handler
			return decorator

		def _register_resource(self, resource_cls, rule: str) -> None:
			methods = []
			for method in ("get", "post", "put", "patch", "delete"):
				if hasattr(resource_cls, method):
					methods.append(method.upper())
			if not methods:
				return

			def view(**kwargs):
				resource = resource_cls()
				handler = getattr(resource, request.method.lower())
				try:
					result = handler(**kwargs)
					if inspect.isawaitable(result):
						try:
							asyncio.get_running_loop()
						except RuntimeError:
							result = asyncio.run(result)
						else:
							previous_loop = events._get_running_loop()
							loop = asyncio.new_event_loop()
							try:
								events._set_running_loop(None)
								result = loop.run_until_complete(result)
							finally:
								pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
								for task in pending:
									task.cancel()
								if pending:
									loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
								loop.close()
								events._set_running_loop(previous_loop)
				except HTTPException as exc:
					response = jsonify({"error": exc.name, "message": exc.description})
					response.status_code = exc.code or 500
					return response
				status = None
				if isinstance(result, tuple):
					result, status = result
				response = jsonify(result)
				if status is not None:
					response.status_code = status
				return response

			endpoint = f"{resource_cls.__name__}_{len(self.blueprint.view_functions)}"
			self.blueprint.add_url_rule(rule, endpoint, view, methods=methods, strict_slashes=False)
from werkzeug.exceptions import BadRequest, NotFound, Unauthorized, Forbidden, HTTPException

from .service import ServiceRegistryService
from .models import (
	ServiceRegistration, ServiceDiscoveryQuery, ServiceDiscoveryResult,
	ServiceHealthStatus, ServiceEvent, ServiceMetrics, ServiceStatus,
	ServiceType, HealthCheckType, CircuitBreakerState, LoadBalanceStrategy
)

# APG Integration Imports
try:
	from ..auth.decorators import require_auth, require_permission
	from ..audl.decorators import audit_log
	APG_AUTH_AVAILABLE = True
except ImportError:
	# Fallback decorators for development
	def require_auth(f): return f
	def require_permission(permission): return lambda f: f
	def audit_log(action): return lambda f: f
	APG_AUTH_AVAILABLE = False

# Create Flask Blueprint
registry_bp = Blueprint('registry', __name__, url_prefix='/api/regy/v1')

# Create Flask-RESTX API
api = Api(
	registry_bp,
	version='1.0.0',
	title='Registry (regy) API',
	description='APG Service Registry API with intelligent discovery and health monitoring',
	doc='/docs/',
	prefix='/api/regy/v1'
)

# API Namespaces
services_ns = Namespace('services', description='Service registration and management')
discovery_ns = Namespace('discovery', description='Intelligent service discovery')  
health_ns = Namespace('health', description='Service health monitoring')
metrics_ns = Namespace('metrics', description='Service metrics and analytics')
events_ns = Namespace('events', description='Registry events and audit logs')

api.add_namespace(services_ns, path='/services')
api.add_namespace(discovery_ns, path='/discovery')
api.add_namespace(health_ns, path='/health')
api.add_namespace(metrics_ns, path='/metrics')
api.add_namespace(events_ns, path='/events')

# API Models for Documentation
service_registration_model = services_ns.model('ServiceRegistration', {
	'name': fields.String(required=True, description='Service name'),
	'display_name': fields.String(required=True, description='Human-readable service name'),
	'description': fields.String(description='Service description'),
	'service_type': fields.String(required=True, description='Type of service'),
	'namespace': fields.String(default='default', description='Service namespace'),
	'environment': fields.String(required=True, description='Deployment environment'),
	'base_path': fields.String(default='/', description='Service base path'),
	'instances': fields.List(fields.Raw, description='Service instances'),
	'versions': fields.List(fields.Raw, description='API versions'),
	'tags': fields.List(fields.String, description='Service tags'),
	'metadata': fields.Raw(description='Custom metadata')
})

service_discovery_query_model = discovery_ns.model('ServiceDiscoveryQuery', {
	'service_name': fields.String(description='Service name filter'),
	'service_type': fields.String(description='Service type filter'),
	'namespace': fields.String(description='Namespace filter'),
	'environment': fields.String(description='Environment filter'),
	'status': fields.String(description='Service status filter'),
	'healthy_only': fields.Boolean(default=True, description='Return only healthy services'),
	'min_health_score': fields.Float(default=0.0, description='Minimum health score'),
	'tags': fields.List(fields.String, description='Required tags'),
	'intelligent_ranking': fields.Boolean(default=False, description='AI-powered ranking'),
	'limit': fields.Integer(default=50, description='Maximum results'),
	'offset': fields.Integer(default=0, description='Result offset')
})

health_status_model = health_ns.model('ServiceHealthStatus', {
	'service_id': fields.String(required=True, description='Service identifier'),
	'instance_id': fields.String(required=True, description='Instance identifier'),
	'overall_status': fields.String(description='Overall health status'),
	'health_score': fields.Float(description='Health score (0-1)'),
	'status_message': fields.String(description='Health status message'),
	'response_time_ms': fields.Float(description='Current response time'),
	'cpu_usage_percent': fields.Float(description='CPU usage percentage'),
	'memory_usage_percent': fields.Float(description='Memory usage percentage'),
	'circuit_breaker_state': fields.String(description='Circuit breaker state')
})

# Global registry service instance
registry_service: Optional[ServiceRegistryService] = None

def get_registry_service() -> ServiceRegistryService:
	"""Get the registry service instance."""
	global registry_service
	if not registry_service:
		# Get tenant ID from request context (APG multi-tenancy)
		tenant_id = get_tenant_id_from_request()
		registry_service = ServiceRegistryService(tenant_id)
	return registry_service

def get_tenant_id_from_request() -> str:
	"""Extract tenant ID from request context."""
	# In production, this would extract from APG auth context
	tenant_id = request.headers.get('X-Tenant-ID')
	if not tenant_id:
		tenant_id = request.args.get('tenant_id', 'default')
	return tenant_id

def get_user_id_from_request() -> str:
	"""Extract user ID from request context."""
	# In production, this would extract from APG auth context
	user_id = request.headers.get('X-User-ID')
	if not user_id:
		user_id = request.args.get('user_id', 'anonymous')
	return user_id

async def ensure_registry_initialized():
	"""Ensure registry service is initialized."""
	service = get_registry_service()
	if not service.initialized:
		await service.initialize()

# Service Registration and Management Endpoints

@services_ns.route('/')
class ServiceList(Resource):
	"""Service registration and listing endpoints."""
	
	@services_ns.doc('list_services')
	@services_ns.marshal_list_with(service_registration_model)
	@require_auth
	@require_permission('registry:list_services')
	@audit_log('list_services')
	async def get(self):
		"""List all registered services."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		# Get query parameters
		namespace = request.args.get('namespace')
		environment = request.args.get('environment')
		service_type = request.args.get('service_type')
		status = request.args.get('status')
		
		# Build discovery query
		query_data = {
			'tenant_id': service.tenant_id,
			'namespace': namespace,
			'environment': environment,
			'service_type': service_type,
			'status': status,
			'limit': int(request.args.get('limit', 100)),
			'offset': int(request.args.get('offset', 0))
		}
		
		# Filter out None values
		query_data = {k: v for k, v in query_data.items() if v is not None}
		
		query = ServiceDiscoveryQuery(**query_data)
		result = await service.discover_services(query)
		
		return {
			'services': [service.model_dump() for service in result.services],
			'total_count': result.total_count,
			'returned_count': result.returned_count,
			'query_time_ms': result.query_time_ms
		}
	
	@services_ns.doc('register_service')
	@services_ns.expect(service_registration_model)
	@services_ns.marshal_with(service_registration_model, code=201)
	@require_auth
	@require_permission('registry:register_service')
	@audit_log('register_service')
	async def post(self):
		"""Register a new service."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		data = request.get_json()
		if not data:
			raise BadRequest("Request body is required")
		
		user_id = get_user_id_from_request()
		
		try:
			registered_service = await service.register_service(data, user_id)
			return registered_service.model_dump(), 201
		except ValueError as e:
			raise BadRequest(str(e))
		except PermissionError as e:
			raise Forbidden(str(e))

@services_ns.route('/<service_id>')
@services_ns.param('service_id', 'Service identifier')
class ServiceDetail(Resource):
	"""Individual service management endpoints."""
	
	@services_ns.doc('get_service')
	@services_ns.marshal_with(service_registration_model)
	@require_auth
	@require_permission('registry:get_service')
	async def get(self, service_id: str):
		"""Get service details by ID."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		if service_id not in service.services:
			raise NotFound(f"Service {service_id} not found")
		
		registered_service = service.services[service_id]
		return registered_service.model_dump()
	
	@services_ns.doc('update_service')
	@services_ns.expect(service_registration_model)
	@services_ns.marshal_with(service_registration_model)
	@require_auth
	@require_permission('registry:update_service')
	@audit_log('update_service')
	async def put(self, service_id: str):
		"""Update service configuration."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		if service_id not in service.services:
			raise NotFound(f"Service {service_id} not found")
		
		data = request.get_json()
		if not data:
			raise BadRequest("Request body is required")
		
		user_id = get_user_id_from_request()
		
		try:
			# Update existing service
			registered_service = service.services[service_id]
			
			# Update fields
			for key, value in data.items():
				if hasattr(registered_service, key):
					setattr(registered_service, key, value)
			
			registered_service.updated_at = datetime.now(timezone.utc)
			registered_service.last_modified_by = user_id
			
			return registered_service.model_dump()
		except ValueError as e:
			raise BadRequest(str(e))
	
	@services_ns.doc('deregister_service')
	@require_auth
	@require_permission('registry:deregister_service')
	@audit_log('deregister_service')
	async def delete(self, service_id: str):
		"""Deregister a service."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		user_id = get_user_id_from_request()
		
		try:
			success = await service.deregister_service(service_id, user_id)
			if not success:
				raise NotFound(f"Service {service_id} not found")
			return {'message': 'Service deregistered successfully'}, 200
		except PermissionError as e:
			raise Forbidden(str(e))

# Service Discovery Endpoints

@discovery_ns.route('/search')
class ServiceDiscovery(Resource):
	"""Intelligent service discovery endpoints."""
	
	@discovery_ns.doc('discover_services')
	@discovery_ns.expect(service_discovery_query_model)
	@require_auth
	@require_permission('registry:discover_services')
	async def post(self):
		"""Discover services with intelligent filtering and ranking."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		data = request.get_json()
		if not data:
			raise BadRequest("Request body is required")
		
		# Add tenant context
		data['tenant_id'] = service.tenant_id
		
		try:
			query = ServiceDiscoveryQuery(**data)
			result = await service.discover_services(query)
			return result.model_dump()
		except ValueError as e:
			raise BadRequest(str(e))

@discovery_ns.route('/by-name/<service_name>')
@discovery_ns.param('service_name', 'Service name')
class ServiceDiscoveryByName(Resource):
	"""Service discovery by name."""
	
	@discovery_ns.doc('discover_by_name')
	@require_auth
	@require_permission('registry:discover_services')
	async def get(self, service_name: str):
		"""Discover services by name."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		# Build query
		query_data = {
			'tenant_id': service.tenant_id,
			'service_name': unquote(service_name),
			'healthy_only': request.args.get('healthy_only', 'true').lower() == 'true',
			'intelligent_ranking': request.args.get('intelligent_ranking', 'false').lower() == 'true',
			'limit': int(request.args.get('limit', 50))
		}
		
		query = ServiceDiscoveryQuery(**query_data)
		result = await service.discover_services(query)
		return result.model_dump()

@discovery_ns.route('/by-type/<service_type>')
@discovery_ns.param('service_type', 'Service type')
class ServiceDiscoveryByType(Resource):
	"""Service discovery by type."""
	
	@discovery_ns.doc('discover_by_type')
	@require_auth
	@require_permission('registry:discover_services')
	async def get(self, service_type: str):
		"""Discover services by type."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		query_data = {
			'tenant_id': service.tenant_id,
			'service_type': unquote(service_type),
			'healthy_only': request.args.get('healthy_only', 'true').lower() == 'true',
			'environment': request.args.get('environment'),
			'namespace': request.args.get('namespace'),
			'limit': int(request.args.get('limit', 50))
		}
		
		# Filter out None values
		query_data = {k: v for k, v in query_data.items() if v is not None}
		
		query = ServiceDiscoveryQuery(**query_data)
		result = await service.discover_services(query)
		return result.model_dump()

# Health Monitoring Endpoints

@health_ns.route('/')
class HealthOverview(Resource):
	"""Registry health overview."""
	
	@health_ns.doc('health_overview')
	@require_auth
	@require_permission('registry:view_health')
	async def get(self):
		"""Get overall registry health status."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		stats = await service.get_registry_statistics()
		return {
			'registry_health': 'healthy' if service.initialized else 'unhealthy',
			'total_services': stats['service_statistics']['total_services'],
			'healthy_services': stats['service_statistics']['healthy_services'],
			'degraded_services': stats['service_statistics']['degraded_services'],
			'unhealthy_services': stats['service_statistics']['unhealthy_services'],
			'uptime_seconds': stats['registry_info']['uptime_seconds'],
			'performance_counters': stats['performance_counters']
		}

@health_ns.route('/services/<service_id>')
@health_ns.param('service_id', 'Service identifier')
class ServiceHealth(Resource):
	"""Service health monitoring."""
	
	@health_ns.doc('get_service_health')
	@health_ns.marshal_with(health_status_model)
	@require_auth
	@require_permission('registry:view_health')
	async def get(self, service_id: str):
		"""Get service health status."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		health_status = await service.get_service_health(service_id)
		if not health_status:
			raise NotFound(f"Health status not found for service {service_id}")
		
		return health_status.model_dump()
	
	@health_ns.doc('update_service_health')
	@health_ns.expect(health_status_model)
	@require_auth
	@require_permission('registry:update_health')
	@audit_log('update_service_health')
	async def put(self, service_id: str):
		"""Update service health status."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		data = request.get_json()
		if not data:
			raise BadRequest("Request body is required")
		
		try:
			success = await service.update_service_health(service_id, data)
			if not success:
				raise NotFound(f"Service {service_id} not found")
			return {'message': 'Health status updated successfully'}, 200
		except ValueError as e:
			raise BadRequest(str(e))

@health_ns.route('/check/<service_id>')
@health_ns.param('service_id', 'Service identifier')
class ServiceHealthCheck(Resource):
	"""Manual health check trigger."""
	
	@health_ns.doc('trigger_health_check')
	@require_auth
	@require_permission('registry:trigger_health_check')
	@audit_log('trigger_health_check')
	async def post(self, service_id: str):
		"""Trigger manual health check for a service."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		if service_id not in service.services:
			raise NotFound(f"Service {service_id} not found")
		
		# Trigger health check (simplified)
		health_status = await service._compute_service_health(service_id)
		service.service_health[f"{service_id}:aggregate"] = health_status
		
		return {
			'message': 'Health check triggered successfully',
			'health_status': health_status.model_dump()
		}

# Metrics and Analytics Endpoints

@metrics_ns.route('/services/<service_id>')
@metrics_ns.param('service_id', 'Service identifier')
class ServiceMetricsEndpoint(Resource):
	"""Service metrics endpoints."""
	
	@metrics_ns.doc('get_service_metrics')
	@require_auth
	@require_permission('registry:view_metrics')
	async def get(self, service_id: str):
		"""Get service performance metrics."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		time_range_hours = int(request.args.get('hours', 24))
		metrics = await service.get_service_metrics(service_id, time_range_hours)
		
		return {
			'service_id': service_id,
			'time_range_hours': time_range_hours,
			'metrics_count': len(metrics),
			'metrics': [metric.model_dump() for metric in metrics]
		}

@metrics_ns.route('/registry/statistics')
class RegistryStatistics(Resource):
	"""Registry statistics and analytics."""
	
	@metrics_ns.doc('get_registry_statistics')
	@require_auth
	@require_permission('registry:view_statistics')
	async def get(self):
		"""Get comprehensive registry statistics."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		stats = await service.get_registry_statistics()
		return stats

# Events and Audit Endpoints

@events_ns.route('/')
class EventsList(Resource):
	"""Registry events listing."""
	
	@events_ns.doc('list_events')
	@require_auth
	@require_permission('registry:view_events')
	async def get(self):
		"""List registry events."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		limit = int(request.args.get('limit', 100))
		offset = int(request.args.get('offset', 0))
		event_type = request.args.get('event_type')
		service_id = request.args.get('service_id')
		severity = request.args.get('severity')
		
		# Filter events
		filtered_events = service.service_events
		
		if event_type:
			filtered_events = [e for e in filtered_events if e.event_type == event_type]
		if service_id:
			filtered_events = [e for e in filtered_events if e.service_id == service_id]
		if severity:
			filtered_events = [e for e in filtered_events if e.severity == severity]
		
		# Keep event order stable for audit readability.
		filtered_events.sort(key=lambda x: x.timestamp)
		
		# Apply pagination
		paginated_events = filtered_events[offset:offset + limit]
		
		return {
			'events': [event.model_dump() for event in paginated_events],
			'total_count': len(filtered_events),
			'returned_count': len(paginated_events)
		}

@events_ns.route('/<event_id>')
@events_ns.param('event_id', 'Event identifier')
class EventDetail(Resource):
	"""Individual event details."""
	
	@events_ns.doc('get_event')
	@require_auth
	@require_permission('registry:view_events')
	async def get(self, event_id: str):
		"""Get event details by ID."""
		await ensure_registry_initialized()
		service = get_registry_service()
		
		event = next((e for e in service.service_events if e.id == event_id), None)
		if not event:
			raise NotFound(f"Event {event_id} not found")
		
		return event.model_dump()

# Registry Management Endpoints

@api.route('/status')
class RegistryStatus(Resource):
	"""Registry system status."""
	
	@api.doc('registry_status')
	async def get(self):
		"""Get registry system status."""
		try:
			await ensure_registry_initialized()
			service = get_registry_service()
			
			stats = await service.get_registry_statistics()
			
			return {
				'status': 'healthy',
				'version': '1.0.0',
				'tenant_id': service.tenant_id,
				'initialized': bool(service.initialized),
				'ml_features_enabled': bool(getattr(service, "ml_models_loaded", False)),
				'apg_integration': APG_AUTH_AVAILABLE,
				'uptime_seconds': stats['registry_info']['uptime_seconds'],
				'performance': stats['performance_counters']
			}
		except Exception as e:
			return {
				'status': 'error',
				'error': str(e),
				'version': '1.0.0'
			}, 500

@api.route('/ready')
class RegistryReadiness(Resource):
	"""Registry readiness probe."""
	
	@api.doc('registry_readiness')
	async def get(self):
		"""Check if registry is ready to serve requests."""
		try:
			service = get_registry_service()
			
			if not service.initialized:
				await service.initialize()
			
			return {
				'ready': True,
				'message': 'Registry is ready to serve requests'
			}
		except Exception as e:
			return {
				'ready': False,
				'error': str(e)
			}, 503

# WebSocket Support for Real-time Updates (Placeholder)
# In production, this would integrate with APG real_time_collaboration

@api.route('/ws/health')
class HealthWebSocket(Resource):
	"""WebSocket endpoint for real-time health updates."""
	
	@api.doc('health_websocket')
	async def get(self):
		"""WebSocket endpoint for real-time service health updates."""
		return {
			'message': 'WebSocket endpoint for real-time health monitoring',
			'endpoint': '/api/regy/v1/ws/health',
			'protocols': ['health-monitoring-v1'],
			'description': 'Connect via WebSocket for live service health updates'
		}

@api.route('/ws/events')  
class EventsWebSocket(Resource):
	"""WebSocket endpoint for real-time event stream."""
	
	@api.doc('events_websocket')
	async def get(self):
		"""WebSocket endpoint for real-time registry events."""
		return {
			'message': 'WebSocket endpoint for real-time event streaming',
			'endpoint': '/api/regy/v1/ws/events',
			'protocols': ['registry-events-v1'],
			'description': 'Connect via WebSocket for live registry event stream'
		}

# Error Handlers

@api.errorhandler(BadRequest)
def handle_bad_request(error):
	return {
		'error': 'Bad Request',
		'message': str(error.description),
		'status_code': 400
	}, 400

@api.errorhandler(Unauthorized)
def handle_unauthorized(error):
	return {
		'error': 'Unauthorized',
		'message': 'Authentication required',
		'status_code': 401
	}, 401

@api.errorhandler(Forbidden)
def handle_forbidden(error):
	return {
		'error': 'Forbidden',
		'message': str(error.description),
		'status_code': 403
	}, 403

@api.errorhandler(NotFound)
def handle_not_found(error):
	return {
		'error': 'Not Found',
		'message': str(error.description),
		'status_code': 404
	}, 404

@api.errorhandler(Exception)
def handle_internal_error(error):
	return {
		'error': 'Internal Server Error',
		'message': 'An unexpected error occurred',
		'status_code': 500
	}, 500

# Export the blueprint for APG integration
__all__ = ['registry_bp', 'api']

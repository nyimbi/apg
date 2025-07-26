"""
APG Integration API Management - REST API Endpoints

RESTful API endpoints for external integration with the API Management platform,
including gateway operations, consumer management, and analytics access.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple

from flask import Blueprint, request, jsonify, g
from flask_appbuilder import AppBuilder
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from marshmallow import Schema, fields, ValidationError, validate

from .models import (
	AMAPI, AMEndpoint, AMPolicy, AMConsumer, AMAPIKey, AMSubscription,
	AMDeployment, AMAnalytics, AMUsageRecord, APIConfig, EndpointConfig,
	PolicyConfig, ConsumerConfig, APIKeyConfig, SubscriptionConfig,
	APIStatus, ProtocolType, AuthenticationType, PolicyType, ConsumerStatus
)
from .service import (
	APILifecycleService, ConsumerManagementService, 
	PolicyManagementService, AnalyticsService
)

# =============================================================================
# Marshmallow Schemas for API Serialization
# =============================================================================

class APISchema(Schema):
	"""Schema for API serialization."""
	
	api_id = fields.String(dump_only=True)
	api_name = fields.String(required=True, validate=validate.Length(min=1, max=200))
	api_title = fields.String(required=True, validate=validate.Length(min=1, max=300))
	api_description = fields.String(validate=validate.Length(max=2000))
	version = fields.String(required=True, validate=validate.Length(max=50))
	protocol_type = fields.String(validate=validate.OneOf([pt.value for pt in ProtocolType]))
	base_path = fields.String(required=True, validate=validate.Length(min=1, max=500))
	upstream_url = fields.Url(required=True, validate=validate.Length(min=1, max=1000))
	status = fields.String(validate=validate.OneOf([s.value for s in APIStatus]))
	is_public = fields.Boolean()
	documentation_url = fields.Url(validate=validate.Length(max=1000))
	openapi_spec = fields.Dict()
	timeout_ms = fields.Integer(validate=validate.Range(min=1000, max=300000))
	retry_attempts = fields.Integer(validate=validate.Range(min=0, max=10))
	auth_type = fields.String(validate=validate.OneOf([at.value for at in AuthenticationType]))
	auth_config = fields.Dict()
	default_rate_limit = fields.Integer(validate=validate.Range(min=1))
	category = fields.String(validate=validate.Length(max=100))
	tags = fields.List(fields.String())
	tenant_id = fields.String(dump_only=True)
	created_at = fields.DateTime(dump_only=True)
	updated_at = fields.DateTime(dump_only=True)
	created_by = fields.String(dump_only=True)
	updated_by = fields.String(dump_only=True)

class EndpointSchema(Schema):
	"""Schema for endpoint serialization."""
	
	endpoint_id = fields.String(dump_only=True)
	api_id = fields.String(required=True)
	path = fields.String(required=True, validate=validate.Length(min=1, max=500))
	method = fields.String(required=True, validate=validate.OneOf(['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']))
	operation_id = fields.String(validate=validate.Length(max=200))
	summary = fields.String(validate=validate.Length(max=300))
	description = fields.String(validate=validate.Length(max=2000))
	request_schema = fields.Dict()
	response_schema = fields.Dict()
	parameters = fields.List(fields.Dict())
	auth_required = fields.Boolean()
	scopes_required = fields.List(fields.String())
	rate_limit_override = fields.Integer(validate=validate.Range(min=1))
	cache_enabled = fields.Boolean()
	cache_ttl_seconds = fields.Integer(validate=validate.Range(min=1))
	deprecated = fields.Boolean()
	examples = fields.Dict()
	created_at = fields.DateTime(dump_only=True)
	updated_at = fields.DateTime(dump_only=True)

class PolicySchema(Schema):
	"""Schema for policy serialization."""
	
	policy_id = fields.String(dump_only=True)
	api_id = fields.String(required=True)
	policy_name = fields.String(required=True, validate=validate.Length(min=1, max=200))
	policy_type = fields.String(required=True, validate=validate.OneOf([pt.value for pt in PolicyType]))
	policy_description = fields.String(validate=validate.Length(max=2000))
	config = fields.Dict(required=True)
	execution_order = fields.Integer(validate=validate.Range(min=0, max=1000))
	enabled = fields.Boolean()
	conditions = fields.Dict()
	applies_to_endpoints = fields.List(fields.String())
	created_at = fields.DateTime(dump_only=True)
	updated_at = fields.DateTime(dump_only=True)
	created_by = fields.String(dump_only=True)

class ConsumerSchema(Schema):
	"""Schema for consumer serialization."""
	
	consumer_id = fields.String(dump_only=True)
	consumer_name = fields.String(required=True, validate=validate.Length(min=1, max=200))
	organization = fields.String(validate=validate.Length(max=300))
	contact_email = fields.Email(required=True, validate=validate.Length(max=255))
	contact_name = fields.String(validate=validate.Length(max=200))
	status = fields.String(validate=validate.OneOf([cs.value for cs in ConsumerStatus]))
	approval_date = fields.DateTime(dump_only=True)
	approved_by = fields.String(dump_only=True)
	allowed_apis = fields.List(fields.String())
	ip_whitelist = fields.List(fields.String())
	global_rate_limit = fields.Integer(validate=validate.Range(min=1))
	global_quota_limit = fields.Integer(validate=validate.Range(min=1))
	portal_access = fields.Boolean()
	tenant_id = fields.String(dump_only=True)
	created_at = fields.DateTime(dump_only=True)
	updated_at = fields.DateTime(dump_only=True)
	created_by = fields.String(dump_only=True)

class APIKeySchema(Schema):
	"""Schema for API key serialization."""
	
	key_id = fields.String(dump_only=True)
	consumer_id = fields.String(required=True)
	key_name = fields.String(required=True, validate=validate.Length(min=1, max=200))
	key_prefix = fields.String(dump_only=True)
	scopes = fields.List(fields.String())
	allowed_apis = fields.List(fields.String())
	active = fields.Boolean()
	expires_at = fields.DateTime()
	last_used_at = fields.DateTime(dump_only=True)
	rate_limit_override = fields.Integer(validate=validate.Range(min=1))
	quota_limit_override = fields.Integer(validate=validate.Range(min=1))
	ip_restrictions = fields.List(fields.String())
	referer_restrictions = fields.List(fields.String())
	created_at = fields.DateTime(dump_only=True)
	updated_at = fields.DateTime(dump_only=True)
	created_by = fields.String(dump_only=True)

class AnalyticsSchema(Schema):
	"""Schema for analytics data serialization."""
	
	metric_name = fields.String()
	metric_value = fields.Float()
	timestamp = fields.DateTime()
	dimensions = fields.Dict()

# =============================================================================
# API Management Endpoints
# =============================================================================

class APIManagementApi(BaseApi):
	"""API endpoints for API lifecycle management."""
	
	resource_name = "apis"
	datamodel = SQLAInterface(AMAPI)
	
	class_permission_name = "APIManagementApi"
	method_permission_name = {
		'get_list': 'read',
		'get': 'read',
		'post': 'create',
		'put': 'update',
		'delete': 'delete'
	}
	
	# Serialization schemas
	add_model_schema = APISchema()
	edit_model_schema = APISchema()
	show_model_schema = APISchema()
	list_model_schema = APISchema()
	
	@expose('/', methods=['GET'])
	@protect()
	def get_list(self):
		"""Get list of APIs with filtering and pagination."""
		
		try:
			# Get query parameters
			page = request.args.get('page', 1, type=int)
			page_size = request.args.get('page_size', 20, type=int)
			search = request.args.get('search', '')
			status = request.args.get('status', '')
			category = request.args.get('category', '')
			is_public = request.args.get('is_public', '')
			
			# Build query
			query = self.datamodel.session.query(AMAPI)
			
			# Apply tenant filter
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			query = query.filter(AMAPI.tenant_id == tenant_id)
			
			# Apply filters
			if search:
				query = query.filter(
					AMAPI.api_name.ilike(f'%{search}%') |
					AMAPI.api_title.ilike(f'%{search}%')
				)
			
			if status:
				query = query.filter(AMAPI.status == status)
			
			if category:
				query = query.filter(AMAPI.category == category)
			
			if is_public:
				query = query.filter(AMAPI.is_public == (is_public.lower() == 'true'))
			
			# Get total count
			total_count = query.count()
			
			# Apply pagination
			offset = (page - 1) * page_size
			apis = query.offset(offset).limit(page_size).all()
			
			# Serialize results
			result = self.list_model_schema.dump(apis, many=True)
			
			return jsonify({
				'success': True,
				'data': result,
				'pagination': {
					'page': page,
					'page_size': page_size,
					'total_count': total_count,
					'total_pages': (total_count + page_size - 1) // page_size
				}
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/<api_id>', methods=['GET'])
	@protect()
	def get(self, api_id: str):
		"""Get specific API details."""
		
		try:
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			
			api = self.datamodel.session.query(AMAPI).filter(
				AMAPI.api_id == api_id,
				AMAPI.tenant_id == tenant_id
			).first()
			
			if not api:
				return jsonify({
					'success': False,
					'error': 'API not found'
				}), 404
			
			result = self.show_model_schema.dump(api)
			
			return jsonify({
				'success': True,
				'data': result
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/', methods=['POST'])
	@protect()
	def post(self):
		"""Create new API."""
		
		try:
			json_data = request.get_json()
			if not json_data:
				return jsonify({
					'success': False,
					'error': 'No JSON data provided'
				}), 400
			
			# Validate input
			try:
				validated_data = self.add_model_schema.load(json_data)
			except ValidationError as err:
				return jsonify({
					'success': False,
					'error': 'Validation error',
					'details': err.messages
				}), 400
			
			# Use service to create API
			api_service = APILifecycleService()
			
			api_config = APIConfig(**validated_data)
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			user_id = g.user.username
			
			api_id = await api_service.register_api(
				config=api_config,
				tenant_id=tenant_id,
				created_by=user_id
			)
			
			# Get created API
			api = self.datamodel.session.query(AMAPI).filter_by(api_id=api_id).first()
			result = self.show_model_schema.dump(api)
			
			return jsonify({
				'success': True,
				'data': result,
				'message': 'API created successfully'
			}), 201
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/<api_id>', methods=['PUT'])
	@protect()
	def put(self, api_id: str):
		"""Update existing API."""
		
		try:
			json_data = request.get_json()
			if not json_data:
				return jsonify({
					'success': False,
					'error': 'No JSON data provided'
				}), 400
			
			# Validate input
			try:
				validated_data = self.edit_model_schema.load(json_data)
			except ValidationError as err:
				return jsonify({
					'success': False,
					'error': 'Validation error',
					'details': err.messages
				}), 400
			
			# Check if API exists
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			api = self.datamodel.session.query(AMAPI).filter(
				AMAPI.api_id == api_id,
				AMAPI.tenant_id == tenant_id
			).first()
			
			if not api:
				return jsonify({
					'success': False,
					'error': 'API not found'
				}), 404
			
			# Use service to update API
			api_service = APILifecycleService()
			
			await api_service.update_api_configuration(
				api_id=api_id,
				updates=validated_data,
				tenant_id=tenant_id,
				updated_by=g.user.username
			)
			
			# Get updated API
			api = self.datamodel.session.query(AMAPI).filter_by(api_id=api_id).first()
			result = self.show_model_schema.dump(api)
			
			return jsonify({
				'success': True,
				'data': result,
				'message': 'API updated successfully'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/<api_id>', methods=['DELETE'])
	@protect()
	def delete(self, api_id: str):
		"""Delete API."""
		
		try:
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			
			# Check if API exists
			api = self.datamodel.session.query(AMAPI).filter(
				AMAPI.api_id == api_id,
				AMAPI.tenant_id == tenant_id
			).first()
			
			if not api:
				return jsonify({
					'success': False,
					'error': 'API not found'
				}), 404
			
			# Use service to delete API
			api_service = APILifecycleService()
			
			await api_service.deregister_api(
				api_id=api_id,
				tenant_id=tenant_id,
				deactivated_by=g.user.username
			)
			
			return jsonify({
				'success': True,
				'message': 'API deleted successfully'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/<api_id>/activate', methods=['POST'])
	@protect()
	def activate_api(self, api_id: str):
		"""Activate API for production use."""
		
		try:
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			
			api_service = APILifecycleService()
			
			await api_service.activate_api(
				api_id=api_id,
				tenant_id=tenant_id,
				activated_by=g.user.username
			)
			
			return jsonify({
				'success': True,
				'message': 'API activated successfully'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/<api_id>/deprecate', methods=['POST'])
	@protect()
	def deprecate_api(self, api_id: str):
		"""Deprecate API with migration timeline."""
		
		try:
			json_data = request.get_json() or {}
			migration_timeline = json_data.get('migration_timeline')
			
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			
			api_service = APILifecycleService()
			
			await api_service.deprecate_api(
				api_id=api_id,
				migration_timeline=migration_timeline,
				tenant_id=tenant_id,
				deprecated_by=g.user.username
			)
			
			return jsonify({
				'success': True,
				'message': 'API deprecated successfully'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

# =============================================================================
# Consumer Management Endpoints
# =============================================================================

class ConsumerManagementApi(BaseApi):
	"""API endpoints for consumer management."""
	
	resource_name = "consumers"
	datamodel = SQLAInterface(AMConsumer)
	
	# Serialization schemas
	add_model_schema = ConsumerSchema()
	edit_model_schema = ConsumerSchema()
	show_model_schema = ConsumerSchema()
	list_model_schema = ConsumerSchema()
	
	@expose('/', methods=['POST'])
	@protect()
	def post(self):
		"""Register new API consumer."""
		
		try:
			json_data = request.get_json()
			if not json_data:
				return jsonify({
					'success': False,
					'error': 'No JSON data provided'
				}), 400
			
			# Validate input
			try:
				validated_data = self.add_model_schema.load(json_data)
			except ValidationError as err:
				return jsonify({
					'success': False,
					'error': 'Validation error',
					'details': err.messages
				}), 400
			
			# Use service to register consumer
			consumer_service = ConsumerManagementService()
			
			consumer_config = ConsumerConfig(**validated_data)
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			
			consumer_id = await consumer_service.register_consumer(
				config=consumer_config,
				tenant_id=tenant_id,
				created_by=g.user.username
			)
			
			# Get created consumer
			consumer = self.datamodel.session.query(AMConsumer).filter_by(consumer_id=consumer_id).first()
			result = self.show_model_schema.dump(consumer)
			
			return jsonify({
				'success': True,
				'data': result,
				'message': 'Consumer registered successfully'
			}), 201
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/<consumer_id>/approve', methods=['POST'])
	@protect()
	def approve_consumer(self, consumer_id: str):
		"""Approve consumer for API access."""
		
		try:
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			
			consumer_service = ConsumerManagementService()
			
			await consumer_service.approve_consumer(
				consumer_id=consumer_id,
				tenant_id=tenant_id,
				approved_by=g.user.username
			)
			
			return jsonify({
				'success': True,
				'message': 'Consumer approved successfully'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/<consumer_id>/api-keys', methods=['POST'])
	@protect()
	def create_api_key(self, consumer_id: str):
		"""Create API key for consumer."""
		
		try:
			json_data = request.get_json()
			if not json_data:
				return jsonify({
					'success': False,
					'error': 'No JSON data provided'
				}), 400
			
			# Validate input
			api_key_schema = APIKeySchema()
			try:
				validated_data = api_key_schema.load(json_data)
			except ValidationError as err:
				return jsonify({
					'success': False,
					'error': 'Validation error',
					'details': err.messages
				}), 400
			
			# Use service to create API key
			consumer_service = ConsumerManagementService()
			
			validated_data['consumer_id'] = consumer_id
			api_key_config = APIKeyConfig(**validated_data)
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			
			api_key_id, api_key = await consumer_service.generate_api_key(
				config=api_key_config,
				tenant_id=tenant_id,
				created_by=g.user.username
			)
			
			return jsonify({
				'success': True,
				'data': {
					'key_id': api_key_id,
					'api_key': api_key,  # Only returned once
					'key_prefix': api_key[:8] + '...'
				},
				'message': 'API key created successfully'
			}), 201
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

# =============================================================================
# Analytics Endpoints
# =============================================================================

class AnalyticsApi(BaseApi):
	"""API endpoints for analytics and monitoring."""
	
	resource_name = "analytics"
	
	@expose('/metrics', methods=['GET'])
	@protect()
	def get_metrics(self):
		"""Get analytics metrics."""
		
		try:
			# Get query parameters
			start_time_str = request.args.get('start_time')
			end_time_str = request.args.get('end_time')
			metric_type = request.args.get('metric_type', 'requests')
			api_id = request.args.get('api_id')
			consumer_id = request.args.get('consumer_id')
			granularity = request.args.get('granularity', '1h')
			
			# Parse time range
			if start_time_str:
				start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
			else:
				start_time = datetime.now(timezone.utc) - timedelta(hours=24)
			
			if end_time_str:
				end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
			else:
				end_time = datetime.now(timezone.utc)
			
			# Get analytics service
			analytics_service = AnalyticsService()
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			
			# Get metrics based on type
			if metric_type == 'requests':
				data = await analytics_service.get_request_volume_over_time(
					start_time=start_time,
					end_time=end_time,
					tenant_id=tenant_id,
					api_id=api_id,
					granularity=granularity
				)
			elif metric_type == 'response_time':
				data = await analytics_service.get_response_time_percentiles(
					start_time=start_time,
					end_time=end_time,
					tenant_id=tenant_id,
					api_id=api_id
				)
			elif metric_type == 'errors':
				data = await analytics_service.get_error_distribution(
					start_time=start_time,
					end_time=end_time,
					tenant_id=tenant_id,
					api_id=api_id
				)
			elif metric_type == 'top_apis':
				data = await analytics_service.get_top_apis_by_usage(
					start_time=start_time,
					end_time=end_time,
					tenant_id=tenant_id,
					limit=10
				)
			else:
				return jsonify({
					'success': False,
					'error': f'Unknown metric type: {metric_type}'
				}), 400
			
			return jsonify({
				'success': True,
				'data': data,
				'metric_type': metric_type,
				'time_range': {
					'start_time': start_time.isoformat(),
					'end_time': end_time.isoformat()
				}
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/summary', methods=['GET'])
	@protect()
	def get_summary(self):
		"""Get analytics summary dashboard."""
		
		try:
			# Get time range (default to last 24 hours)
			end_time = datetime.now(timezone.utc)
			start_time = end_time - timedelta(hours=24)
			
			analytics_service = AnalyticsService()
			tenant_id = getattr(g.user, 'tenant_id', 'default')
			
			# Get summary metrics
			total_requests = await analytics_service.get_total_requests(
				start_time=start_time,
				end_time=end_time,
				tenant_id=tenant_id
			)
			
			avg_response_time = await analytics_service.get_average_response_time(
				start_time=start_time,
				end_time=end_time,
				tenant_id=tenant_id
			)
			
			error_rate = await analytics_service.get_error_rate(
				start_time=start_time,
				end_time=end_time,
				tenant_id=tenant_id
			)
			
			active_consumers = await analytics_service.get_active_consumers_count(
				start_time=start_time,
				end_time=end_time,
				tenant_id=tenant_id
			)
			
			return jsonify({
				'success': True,
				'data': {
					'total_requests': total_requests,
					'avg_response_time_ms': round(avg_response_time, 2) if avg_response_time else 0,
					'error_rate_percent': round(error_rate * 100, 2) if error_rate else 0,
					'active_consumers': active_consumers,
					'time_period_hours': 24
				}
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

# =============================================================================
# Gateway Runtime Endpoints
# =============================================================================

class GatewayApi(BaseApi):
	"""API endpoints for gateway runtime operations."""
	
	resource_name = "gateway"
	
	@expose('/proxy/<path:api_path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
	def proxy_request(self, api_path: str):
		"""Proxy API requests through the gateway."""
		
		try:
			# This would be the main gateway proxy logic
			# For now, return a placeholder response
			
			return jsonify({
				'message': 'Gateway proxy endpoint',
				'path': api_path,
				'method': request.method,
				'headers': dict(request.headers),
				'note': 'This endpoint would proxy to upstream services'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/health', methods=['GET'])
	def health_check(self):
		"""Gateway health check."""
		
		return jsonify({
			'status': 'healthy',
			'timestamp': datetime.now(timezone.utc).isoformat(),
			'version': '1.0.0',
			'component': 'api_gateway'
		})

# =============================================================================
# Blueprint and Registration
# =============================================================================

def register_api_endpoints(appbuilder: AppBuilder):
	"""Register all API endpoints with Flask-AppBuilder."""
	
	# API Management endpoints
	appbuilder.add_api(APIManagementApi)
	
	# Consumer Management endpoints
	appbuilder.add_api(ConsumerManagementApi)
	
	# Analytics endpoints
	appbuilder.add_api(AnalyticsApi)
	
	# Gateway runtime endpoints
	appbuilder.add_api(GatewayApi)

# Export for use in blueprint
__all__ = [
	'APIManagementApi',
	'ConsumerManagementApi', 
	'AnalyticsApi',
	'GatewayApi',
	'register_api_endpoints'
]
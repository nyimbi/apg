"""
APG Integration API Management - Flask-AppBuilder Views

Comprehensive UI views for API gateway management, consumer management, 
analytics dashboards, and developer portal with real-time monitoring.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

from flask import flash, request, redirect, url_for, jsonify, render_template_string
from flask_appbuilder import ModelView, ModelRestApi, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, FormWidget
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.security.decorators import protect
from wtforms import StringField, TextAreaField, SelectField, IntegerField, BooleanField, FormField
from wtforms.validators import DataRequired, Length, NumberRange, Email, URL
from wtforms.widgets import TextArea

from .models import (
	AMAPI, AMEndpoint, AMPolicy, AMConsumer, AMAPIKey, AMSubscription, 
	AMDeployment, AMAnalytics, AMUsageRecord,
	APIStatus, ProtocolType, AuthenticationType, PolicyType, 
	DeploymentStrategy, ConsumerStatus, LoadBalancingAlgorithm
)
from .service import (
	APILifecycleService, ConsumerManagementService, 
	PolicyManagementService, AnalyticsService
)

# =============================================================================
# Custom Widgets and Forms
# =============================================================================

class JSONTextAreaWidget(TextArea):
	"""Custom widget for JSON editing with syntax highlighting."""
	
	def __call__(self, field, **kwargs):
		kwargs.setdefault('class_', 'form-control json-editor')
		kwargs.setdefault('rows', 10)
		kwargs.setdefault('data-mode', 'json')
		return super().__call__(field, **kwargs)

class APIConfigForm(DynamicForm):
	"""Form for API configuration."""
	
	api_name = StringField(
		'API Name',
		validators=[DataRequired(), Length(min=1, max=200)],
		description='Unique name for the API'
	)
	
	api_title = StringField(
		'API Title',
		validators=[DataRequired(), Length(min=1, max=300)],
		description='Human-readable title for the API'
	)
	
	api_description = TextAreaField(
		'Description',
		validators=[Length(max=2000)],
		description='Detailed description of the API functionality'
	)
	
	version = StringField(
		'Version',
		default='1.0.0',
		validators=[DataRequired(), Length(max=50)],
		description='Semantic version (e.g., 1.0.0)'
	)
	
	protocol_type = SelectField(
		'Protocol Type',
		choices=[(pt.value, pt.value.upper()) for pt in ProtocolType],
		default=ProtocolType.REST.value,
		validators=[DataRequired()]
	)
	
	base_path = StringField(
		'Base Path',
		validators=[DataRequired(), Length(min=1, max=500)],
		description='Base URL path (must start with /)'
	)
	
	upstream_url = StringField(
		'Upstream URL',
		validators=[DataRequired(), URL(), Length(min=1, max=1000)],
		description='Backend service URL'
	)
	
	is_public = BooleanField(
		'Public API',
		default=False,
		description='Allow public access without authentication'
	)
	
	documentation_url = StringField(
		'Documentation URL',
		validators=[URL(), Length(max=1000)],
		description='Link to API documentation'
	)
	
	timeout_ms = IntegerField(
		'Timeout (ms)',
		default=30000,
		validators=[NumberRange(min=1000, max=300000)],
		description='Request timeout in milliseconds'
	)
	
	retry_attempts = IntegerField(
		'Retry Attempts',
		default=3,
		validators=[NumberRange(min=0, max=10)],
		description='Number of retry attempts on failure'
	)
	
	load_balancing_algorithm = SelectField(
		'Load Balancing',
		choices=[(alg.value, alg.value.replace('_', ' ').title()) for alg in LoadBalancingAlgorithm],
		default=LoadBalancingAlgorithm.ROUND_ROBIN.value
	)
	
	auth_type = SelectField(
		'Authentication Type',
		choices=[(auth.value, auth.value.replace('_', ' ').title()) for auth in AuthenticationType],
		default=AuthenticationType.API_KEY.value
	)
	
	auth_config = TextAreaField(
		'Auth Configuration',
		widget=JSONTextAreaWidget(),
		description='Authentication configuration (JSON format)'
	)
	
	default_rate_limit = IntegerField(
		'Default Rate Limit',
		validators=[NumberRange(min=1)],
		description='Default requests per minute'
	)
	
	category = StringField(
		'Category',
		validators=[Length(max=100)],
		description='API category for organization'
	)
	
	tags = TextAreaField(
		'Tags',
		description='Comma-separated tags for search and filtering'
	)

class PolicyConfigForm(DynamicForm):
	"""Form for policy configuration."""
	
	policy_name = StringField(
		'Policy Name',
		validators=[DataRequired(), Length(min=1, max=200)]
	)
	
	policy_type = SelectField(
		'Policy Type',
		choices=[(pt.value, pt.value.replace('_', ' ').title()) for pt in PolicyType],
		validators=[DataRequired()]
	)
	
	policy_description = TextAreaField(
		'Description',
		validators=[Length(max=2000)]
	)
	
	config = TextAreaField(
		'Configuration',
		validators=[DataRequired()],
		widget=JSONTextAreaWidget(),
		description='Policy configuration in JSON format'
	)
	
	execution_order = IntegerField(
		'Execution Order',
		default=100,
		validators=[NumberRange(min=0, max=1000)],
		description='Lower numbers execute first'
	)
	
	enabled = BooleanField(
		'Enabled',
		default=True
	)
	
	conditions = TextAreaField(
		'Conditions',
		widget=JSONTextAreaWidget(),
		description='Conditions for policy execution (JSON format)'
	)
	
	applies_to_endpoints = TextAreaField(
		'Applies to Endpoints',
		description='Endpoint IDs (one per line, empty for all endpoints)'
	)

class ConsumerRegistrationForm(DynamicForm):
	"""Form for consumer registration."""
	
	consumer_name = StringField(
		'Consumer Name',
		validators=[DataRequired(), Length(min=1, max=200)]
	)
	
	organization = StringField(
		'Organization',
		validators=[Length(max=300)]
	)
	
	contact_email = StringField(
		'Contact Email',
		validators=[DataRequired(), Email(), Length(max=255)]
	)
	
	contact_name = StringField(
		'Contact Name',
		validators=[Length(max=200)]
	)
	
	allowed_apis = TextAreaField(
		'Allowed APIs',
		description='API IDs (one per line, empty for all public APIs)'
	)
	
	ip_whitelist = TextAreaField(
		'IP Whitelist',
		description='Allowed IP addresses (one per line)'
	)
	
	global_rate_limit = IntegerField(
		'Global Rate Limit',
		validators=[NumberRange(min=1)],
		description='Maximum requests per minute across all APIs'
	)
	
	global_quota_limit = IntegerField(
		'Global Quota Limit',
		validators=[NumberRange(min=1)],
		description='Maximum requests per month'
	)
	
	portal_access = BooleanField(
		'Developer Portal Access',
		default=True
	)

# =============================================================================
# API Management Views
# =============================================================================

class APIManagementView(ModelView):
	"""Main view for API management."""
	
	datamodel = SQLAInterface(AMAPI)
	
	list_title = "API Management"
	show_title = "API Details"
	add_title = "Register New API"
	edit_title = "Edit API"
	
	list_columns = [
		'api_name', 'api_title', 'version', 'protocol_type', 
		'status', 'is_public', 'created_at', 'updated_at'
	]
	
	show_columns = [
		'api_id', 'api_name', 'api_title', 'api_description', 'version',
		'protocol_type', 'base_path', 'upstream_url', 'status', 'is_public',
		'documentation_url', 'timeout_ms', 'retry_attempts', 
		'load_balancing_algorithm', 'auth_type', 'auth_config',
		'default_rate_limit', 'category', 'tags', 'tenant_id',
		'created_at', 'updated_at', 'created_by', 'updated_by'
	]
	
	add_columns = [
		'api_name', 'api_title', 'api_description', 'version',
		'protocol_type', 'base_path', 'upstream_url', 'is_public',
		'documentation_url', 'timeout_ms', 'retry_attempts',
		'load_balancing_algorithm', 'auth_type', 'auth_config',
		'default_rate_limit', 'category', 'tags'
	]
	
	edit_columns = add_columns
	
	search_columns = ['api_name', 'api_title', 'category', 'tags']
	
	list_filters = ['status', 'protocol_type', 'is_public', 'auth_type', 'category']
	
	order_columns = ['api_name', 'version', 'created_at', 'updated_at']
	
	page_size = 20
	
	add_form = APIConfigForm
	edit_form = APIConfigForm
	
	show_fieldsets = [
		('Basic Information', {
			'fields': ['api_id', 'api_name', 'api_title', 'api_description', 'version']
		}),
		('Technical Configuration', {
			'fields': ['protocol_type', 'base_path', 'upstream_url', 'timeout_ms', 'retry_attempts']
		}),
		('Security & Access', {
			'fields': ['auth_type', 'auth_config', 'is_public', 'default_rate_limit']
		}),
		('Operational', {
			'fields': ['status', 'load_balancing_algorithm', 'category', 'tags']
		}),
		('Metadata', {
			'fields': ['tenant_id', 'created_at', 'updated_at', 'created_by', 'updated_by']
		})
	]
	
	add_fieldsets = [
		('Basic Information', {
			'fields': ['api_name', 'api_title', 'api_description', 'version']
		}),
		('Technical Configuration', {
			'fields': ['protocol_type', 'base_path', 'upstream_url', 'timeout_ms', 'retry_attempts']
		}),
		('Security & Access', {
			'fields': ['auth_type', 'auth_config', 'is_public', 'default_rate_limit']
		}),
		('Operational', {
			'fields': ['load_balancing_algorithm', 'category', 'tags']
		}),
		('Documentation', {
			'fields': ['documentation_url']
		})
	]
	
	edit_fieldsets = add_fieldsets
	
	formatters_columns = {
		'tags': lambda x: ', '.join(x.tags) if x.tags else '',
		'auth_config': lambda x: json.dumps(x.auth_config, indent=2) if x.auth_config else '{}',
		'status': lambda x: f'<span class="label label-{self._get_status_class(x.status)}">{x.status.title()}</span>'
	}
	
	@staticmethod
	def _get_status_class(status: str) -> str:
		"""Get Bootstrap label class for status."""
		status_classes = {
			'active': 'success',
			'draft': 'warning',
			'deprecated': 'warning',
			'retired': 'danger',
			'blocked': 'danger'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Pre-process before adding new API."""
		# Set tenant and user info
		item.tenant_id = g.user.tenant_id if hasattr(g.user, 'tenant_id') else 'default'
		item.created_by = g.user.username
		
		# Process tags
		if hasattr(item, 'tags') and isinstance(item.tags, str):
			item.tags = [tag.strip() for tag in item.tags.split(',') if tag.strip()]
		
		# Process auth_config
		if hasattr(item, 'auth_config') and isinstance(item.auth_config, str):
			try:
				item.auth_config = json.loads(item.auth_config)
			except json.JSONDecodeError:
				item.auth_config = {}
	
	def pre_update(self, item):
		"""Pre-process before updating API."""
		item.updated_by = g.user.username
		self.pre_add(item)  # Reuse preprocessing logic

class EndpointManagementView(ModelView):
	"""View for managing API endpoints."""
	
	datamodel = SQLAInterface(AMEndpoint)
	
	list_title = "API Endpoints"
	show_title = "Endpoint Details"
	add_title = "Add Endpoint"
	edit_title = "Edit Endpoint"
	
	list_columns = [
		'api.api_name', 'path', 'method', 'summary', 
		'auth_required', 'deprecated', 'created_at'
	]
	
	show_columns = [
		'endpoint_id', 'api', 'path', 'method', 'operation_id', 'summary',
		'description', 'auth_required', 'scopes_required', 'rate_limit_override',
		'cache_enabled', 'cache_ttl_seconds', 'deprecated', 'request_schema',
		'response_schema', 'parameters', 'examples', 'created_at', 'updated_at'
	]
	
	add_columns = [
		'api', 'path', 'method', 'operation_id', 'summary', 'description',
		'auth_required', 'scopes_required', 'rate_limit_override',
		'cache_enabled', 'cache_ttl_seconds', 'deprecated'
	]
	
	edit_columns = add_columns
	
	search_columns = ['path', 'method', 'summary', 'operation_id']
	
	list_filters = ['api', 'method', 'auth_required', 'deprecated', 'cache_enabled']
	
	related_views = [APIManagementView]
	
	formatters_columns = {
		'scopes_required': lambda x: ', '.join(x.scopes_required) if x.scopes_required else '',
		'method': lambda x: f'<span class="label label-{self._get_method_class(x.method)}">{x.method}</span>'
	}
	
	@staticmethod
	def _get_method_class(method: str) -> str:
		"""Get Bootstrap label class for HTTP method."""
		method_classes = {
			'GET': 'info',
			'POST': 'success',
			'PUT': 'warning',
			'DELETE': 'danger',
			'PATCH': 'warning',
			'HEAD': 'default',
			'OPTIONS': 'default'
		}
		return method_classes.get(method, 'default')

class PolicyManagementView(ModelView):
	"""View for managing API policies."""
	
	datamodel = SQLAInterface(AMPolicy)
	
	list_title = "API Policies"
	show_title = "Policy Details"
	add_title = "Create Policy"
	edit_title = "Edit Policy"
	
	list_columns = [
		'api.api_name', 'policy_name', 'policy_type', 
		'execution_order', 'enabled', 'created_at'
	]
	
	show_columns = [
		'policy_id', 'api', 'policy_name', 'policy_type', 'policy_description',
		'config', 'execution_order', 'enabled', 'conditions', 'applies_to_endpoints',
		'created_at', 'updated_at', 'created_by'
	]
	
	add_columns = [
		'api', 'policy_name', 'policy_type', 'policy_description',
		'config', 'execution_order', 'enabled', 'conditions', 'applies_to_endpoints'
	]
	
	edit_columns = add_columns
	
	search_columns = ['policy_name', 'policy_type', 'policy_description']
	
	list_filters = ['api', 'policy_type', 'enabled']
	
	order_columns = ['execution_order', 'policy_name', 'created_at']
	
	add_form = PolicyConfigForm
	edit_form = PolicyConfigForm
	
	formatters_columns = {
		'config': lambda x: json.dumps(x.config, indent=2) if x.config else '{}',
		'policy_type': lambda x: x.policy_type.replace('_', ' ').title(),
		'enabled': lambda x: '<span class="label label-success">Yes</span>' if x.enabled else '<span class="label label-default">No</span>'
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new policy."""
		item.created_by = g.user.username
		
		# Process JSON fields
		for field in ['config', 'conditions']:
			value = getattr(item, field, None)
			if value and isinstance(value, str):
				try:
					setattr(item, field, json.loads(value))
				except json.JSONDecodeError:
					setattr(item, field, {})
		
		# Process applies_to_endpoints
		if hasattr(item, 'applies_to_endpoints') and isinstance(item.applies_to_endpoints, str):
			item.applies_to_endpoints = [
				ep.strip() for ep in item.applies_to_endpoints.split('\n') 
				if ep.strip()
			]
	
	pre_update = pre_add

# =============================================================================
# Consumer Management Views
# =============================================================================

class ConsumerManagementView(ModelView):
	"""View for managing API consumers."""
	
	datamodel = SQLAInterface(AMConsumer)
	
	list_title = "API Consumers"
	show_title = "Consumer Details"
	add_title = "Register Consumer"
	edit_title = "Edit Consumer"
	
	list_columns = [
		'consumer_name', 'organization', 'contact_email', 
		'status', 'global_rate_limit', 'created_at'
	]
	
	show_columns = [
		'consumer_id', 'consumer_name', 'organization', 'contact_email', 'contact_name',
		'status', 'approval_date', 'approved_by', 'allowed_apis', 'ip_whitelist',
		'global_rate_limit', 'global_quota_limit', 'portal_access', 'tenant_id',
		'created_at', 'updated_at', 'created_by'
	]
	
	add_columns = [
		'consumer_name', 'organization', 'contact_email', 'contact_name',
		'allowed_apis', 'ip_whitelist', 'global_rate_limit', 'global_quota_limit',
		'portal_access'
	]
	
	edit_columns = add_columns + ['status']
	
	search_columns = ['consumer_name', 'organization', 'contact_email']
	
	list_filters = ['status', 'organization', 'portal_access']
	
	add_form = ConsumerRegistrationForm
	edit_form = ConsumerRegistrationForm
	
	formatters_columns = {
		'status': lambda x: f'<span class="label label-{self._get_status_class(x.status)}">{x.status.title()}</span>',
		'allowed_apis': lambda x: f'{len(x.allowed_apis)} APIs' if x.allowed_apis else 'All Public APIs',
		'ip_whitelist': lambda x: f'{len(x.ip_whitelist)} IPs' if x.ip_whitelist else 'Any IP'
	}
	
	@staticmethod
	def _get_status_class(status: str) -> str:
		"""Get Bootstrap label class for consumer status."""
		status_classes = {
			'active': 'success',
			'pending': 'warning',
			'suspended': 'warning',
			'rejected': 'danger'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Pre-process before adding new consumer."""
		item.tenant_id = g.user.tenant_id if hasattr(g.user, 'tenant_id') else 'default'
		item.created_by = g.user.username
		item.status = ConsumerStatus.PENDING.value
		
		# Process list fields
		for field in ['allowed_apis', 'ip_whitelist']:
			value = getattr(item, field, None)
			if value and isinstance(value, str):
				setattr(item, field, [
					line.strip() for line in value.split('\n') 
					if line.strip()
				])
	
	def pre_update(self, item):
		"""Pre-process before updating consumer."""
		self.pre_add(item)
		
		# Handle status changes
		if item.status == ConsumerStatus.ACTIVE.value and not item.approval_date:
			item.approval_date = datetime.now(timezone.utc)
			item.approved_by = g.user.username

class APIKeyManagementView(ModelView):
	"""View for managing API keys."""
	
	datamodel = SQLAInterface(AMAPIKey)
	
	list_title = "API Keys"
	show_title = "API Key Details"
	add_title = "Generate API Key"
	edit_title = "Edit API Key"
	
	list_columns = [
		'consumer.consumer_name', 'key_name', 'key_prefix', 
		'active', 'expires_at', 'last_used_at', 'created_at'
	]
	
	show_columns = [
		'key_id', 'consumer', 'key_name', 'key_prefix', 'scopes',
		'allowed_apis', 'active', 'expires_at', 'last_used_at',
		'rate_limit_override', 'quota_limit_override', 'ip_restrictions',
		'referer_restrictions', 'created_at', 'updated_at', 'created_by'
	]
	
	add_columns = [
		'consumer', 'key_name', 'scopes', 'allowed_apis', 
		'expires_at', 'rate_limit_override', 'quota_limit_override',
		'ip_restrictions', 'referer_restrictions'
	]
	
	edit_columns = add_columns + ['active']
	
	search_columns = ['key_name', 'key_prefix']
	
	list_filters = ['consumer', 'active', 'expires_at']
	
	formatters_columns = {
		'active': lambda x: '<span class="label label-success">Active</span>' if x.active else '<span class="label label-default">Inactive</span>',
		'expires_at': lambda x: x.expires_at.strftime('%Y-%m-%d') if x.expires_at else 'Never',
		'scopes': lambda x: ', '.join(x.scopes) if x.scopes else 'None',
		'key_prefix': lambda x: f'{x.key_prefix}...' if x.key_prefix else ''
	}

# =============================================================================
# Analytics and Monitoring Views
# =============================================================================

class AnalyticsDashboardView(BaseView):
	"""Main analytics dashboard."""
	
	route_base = "/analytics"
	default_view = "dashboard"
	
	@expose("/dashboard/")
	@has_access
	def dashboard(self):
		"""Main analytics dashboard."""
		
		# Get analytics service
		analytics_service = AnalyticsService()
		
		# Get dashboard data
		try:
			# Time range for analytics (last 24 hours)
			end_time = datetime.now(timezone.utc)
			start_time = end_time - timedelta(hours=24)
			
			# Get key metrics
			total_requests = analytics_service.get_total_requests(
				start_time=start_time,
				end_time=end_time,
				tenant_id=g.user.tenant_id if hasattr(g.user, 'tenant_id') else 'default'
			)
			
			avg_response_time = analytics_service.get_average_response_time(
				start_time=start_time,
				end_time=end_time,
				tenant_id=g.user.tenant_id if hasattr(g.user, 'tenant_id') else 'default'
			)
			
			error_rate = analytics_service.get_error_rate(
				start_time=start_time,
				end_time=end_time,
				tenant_id=g.user.tenant_id if hasattr(g.user, 'tenant_id') else 'default'
			)
			
			# Get top APIs
			top_apis = analytics_service.get_top_apis_by_usage(
				start_time=start_time,
				end_time=end_time,
				tenant_id=g.user.tenant_id if hasattr(g.user, 'tenant_id') else 'default',
				limit=10
			)
			
			dashboard_data = {
				'total_requests': total_requests,
				'avg_response_time': round(avg_response_time, 2) if avg_response_time else 0,
				'error_rate': round(error_rate * 100, 2) if error_rate else 0,
				'top_apis': top_apis,
				'time_range': '24 hours'
			}
			
		except Exception as e:
			flash(f'Error loading dashboard data: {str(e)}', 'error')
			dashboard_data = {
				'total_requests': 0,
				'avg_response_time': 0,
				'error_rate': 0,
				'top_apis': [],
				'time_range': '24 hours'
			}
		
		return self.render_template(
			'analytics_dashboard.html',
			dashboard_data=dashboard_data
		)
	
	@expose("/api-usage/")
	@has_access
	def api_usage(self):
		"""API usage analytics."""
		return self.render_template('api_usage_analytics.html')
	
	@expose("/performance/")
	@has_access
	def performance(self):
		"""Performance analytics."""
		return self.render_template('performance_analytics.html')
	
	@expose("/api/metrics/")
	@has_access
	def metrics_api(self):
		"""API endpoint for metrics data."""
		
		# Get query parameters
		metric_type = request.args.get('type', 'requests')
		time_range = request.args.get('range', '1h')
		api_id = request.args.get('api_id')
		
		analytics_service = AnalyticsService()
		
		try:
			# Parse time range
			if time_range == '1h':
				start_time = datetime.now(timezone.utc) - timedelta(hours=1)
			elif time_range == '24h':
				start_time = datetime.now(timezone.utc) - timedelta(hours=24)
			elif time_range == '7d':
				start_time = datetime.now(timezone.utc) - timedelta(days=7)
			else:
				start_time = datetime.now(timezone.utc) - timedelta(hours=1)
			
			end_time = datetime.now(timezone.utc)
			tenant_id = g.user.tenant_id if hasattr(g.user, 'tenant_id') else 'default'
			
			# Get metrics based on type
			if metric_type == 'requests':
				data = analytics_service.get_request_volume_over_time(
					start_time=start_time,
					end_time=end_time,
					tenant_id=tenant_id,
					api_id=api_id,
					granularity='5m'
				)
			elif metric_type == 'response_time':
				data = analytics_service.get_response_time_percentiles(
					start_time=start_time,
					end_time=end_time,
					tenant_id=tenant_id,
					api_id=api_id
				)
			elif metric_type == 'errors':
				data = analytics_service.get_error_distribution(
					start_time=start_time,
					end_time=end_time,
					tenant_id=tenant_id,
					api_id=api_id
				)
			else:
				data = []
			
			return jsonify({
				'success': True,
				'data': data,
				'metric_type': metric_type,
				'time_range': time_range
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

class UsageRecordsView(ModelView):
	"""View for detailed usage records."""
	
	datamodel = SQLAInterface(AMUsageRecord)
	
	list_title = "Usage Records"
	show_title = "Usage Record Details"
	
	list_columns = [
		'timestamp', 'consumer.consumer_name', 'api_id', 'endpoint_path',
		'method', 'response_status', 'response_time_ms', 'billable'
	]
	
	show_columns = [
		'record_id', 'request_id', 'consumer', 'api_id', 'endpoint_path', 'method',
		'timestamp', 'response_status', 'response_time_ms', 'request_size_bytes',
		'response_size_bytes', 'client_ip', 'user_agent', 'referer',
		'country_code', 'region', 'billable', 'cost', 'error_code', 'error_message'
	]
	
	search_columns = ['request_id', 'endpoint_path', 'client_ip']
	
	list_filters = [
		'consumer', 'response_status', 'method', 'billable', 
		'country_code', 'timestamp'
	]
	
	order_columns = ['timestamp', 'response_time_ms', 'response_status']
	
	base_order = ('timestamp', 'desc')
	
	page_size = 50
	
	# Read-only view
	add_exclude_columns = show_columns
	edit_exclude_columns = show_columns
	
	formatters_columns = {
		'response_status': lambda x: f'<span class="label label-{self._get_status_class(x.response_status)}">{x.response_status}</span>',
		'response_time_ms': lambda x: f'{x.response_time_ms}ms',
		'method': lambda x: f'<span class="label label-{self._get_method_class(x.method)}">{x.method}</span>',
		'billable': lambda x: '<span class="label label-success">Yes</span>' if x.billable else '<span class="label label-default">No</span>',
		'request_size_bytes': lambda x: self._format_bytes(x.request_size_bytes) if x.request_size_bytes else '',
		'response_size_bytes': lambda x: self._format_bytes(x.response_size_bytes) if x.response_size_bytes else ''
	}
	
	@staticmethod
	def _get_status_class(status: int) -> str:
		"""Get Bootstrap label class for HTTP status."""
		if 200 <= status < 300:
			return 'success'
		elif 300 <= status < 400:
			return 'info'
		elif 400 <= status < 500:
			return 'warning'
		else:
			return 'danger'
	
	@staticmethod
	def _get_method_class(method: str) -> str:
		"""Get Bootstrap label class for HTTP method."""
		method_classes = {
			'GET': 'info',
			'POST': 'success',
			'PUT': 'warning',
			'DELETE': 'danger',
			'PATCH': 'warning'
		}
		return method_classes.get(method, 'default')
	
	@staticmethod
	def _format_bytes(bytes_value: int) -> str:
		"""Format bytes into human-readable format."""
		if not bytes_value:
			return ''
		
		for unit in ['B', 'KB', 'MB', 'GB']:
			if bytes_value < 1024:
				return f'{bytes_value:.1f} {unit}'
			bytes_value /= 1024
		return f'{bytes_value:.1f} TB'

# =============================================================================
# Developer Portal Views
# =============================================================================

class DeveloperPortalView(BaseView):
	"""Developer portal for API exploration."""
	
	route_base = "/developer"
	default_view = "portal"
	
	@expose("/")
	@expose("/portal/")
	@has_access
	def portal(self):
		"""Main developer portal."""
		
		# Get public APIs
		try:
			from sqlalchemy import and_
			
			public_apis = self.appbuilder.session.query(AMAPI).filter(
				and_(
					AMAPI.is_public == True,
					AMAPI.status == APIStatus.ACTIVE.value,
					AMAPI.tenant_id == (g.user.tenant_id if hasattr(g.user, 'tenant_id') else 'default')
				)
			).all()
			
			# Group APIs by category
			api_categories = {}
			for api in public_apis:
				category = api.category or 'Uncategorized'
				if category not in api_categories:
					api_categories[category] = []
				api_categories[category].append(api)
			
		except Exception as e:
			flash(f'Error loading APIs: {str(e)}', 'error')
			api_categories = {}
		
		return self.render_template(
			'developer_portal.html',
			api_categories=api_categories
		)
	
	@expose("/api/<api_id>/")
	@has_access
	def api_details(self, api_id):
		"""API details and documentation."""
		
		try:
			api = self.appbuilder.session.query(AMAPI).filter_by(api_id=api_id).first()
			
			if not api:
				flash('API not found', 'error')
				return redirect(url_for('DeveloperPortalView.portal'))
			
			# Get endpoints
			endpoints = self.appbuilder.session.query(AMEndpoint).filter_by(api_id=api_id).all()
			
			# Get policies
			policies = self.appbuilder.session.query(AMPolicy).filter_by(api_id=api_id).all()
			
		except Exception as e:
			flash(f'Error loading API details: {str(e)}', 'error')
			return redirect(url_for('DeveloperPortalView.portal'))
		
		return self.render_template(
			'api_documentation.html',
			api=api,
			endpoints=endpoints,
			policies=policies
		)
	
	@expose("/api/<api_id>/try/")
	@has_access
	def try_api(self, api_id):
		"""Interactive API testing interface."""
		
		try:
			api = self.appbuilder.session.query(AMAPI).filter_by(api_id=api_id).first()
			
			if not api:
				flash('API not found', 'error')
				return redirect(url_for('DeveloperPortalView.portal'))
			
			endpoints = self.appbuilder.session.query(AMEndpoint).filter_by(api_id=api_id).all()
			
		except Exception as e:
			flash(f'Error loading API: {str(e)}', 'error')
			return redirect(url_for('DeveloperPortalView.portal'))
		
		return self.render_template(
			'api_testing.html',
			api=api,
			endpoints=endpoints
		)

# =============================================================================
# Deployment Management Views
# =============================================================================

class DeploymentManagementView(ModelView):
	"""View for managing API deployments."""
	
	datamodel = SQLAInterface(AMDeployment)
	
	list_title = "API Deployments"
	show_title = "Deployment Details"
	add_title = "Create Deployment"
	edit_title = "Edit Deployment"
	
	list_columns = [
		'api.api_name', 'deployment_name', 'environment', 'strategy',
		'from_version', 'to_version', 'status', 'progress_percentage', 'started_at'
	]
	
	show_columns = [
		'deployment_id', 'api', 'deployment_name', 'strategy', 'environment',
		'from_version', 'to_version', 'status', 'progress_percentage', 'config',
		'traffic_percentage', 'rollback_available', 'rollback_reason',
		'started_at', 'completed_at', 'created_at', 'created_by'
	]
	
	add_columns = [
		'api', 'deployment_name', 'strategy', 'environment',
		'from_version', 'to_version', 'config', 'traffic_percentage'
	]
	
	edit_columns = add_columns + ['status', 'progress_percentage']
	
	search_columns = ['deployment_name', 'environment']
	
	list_filters = ['api', 'environment', 'strategy', 'status']
	
	formatters_columns = {
		'status': lambda x: f'<span class="label label-{self._get_deployment_status_class(x.status)}">{x.status.title()}</span>',
		'progress_percentage': lambda x: f'{x.progress_percentage}%',
		'traffic_percentage': lambda x: f'{x.traffic_percentage}%',
		'strategy': lambda x: x.strategy.replace('_', ' ').title()
	}
	
	@staticmethod
	def _get_deployment_status_class(status: str) -> str:
		"""Get Bootstrap label class for deployment status."""
		status_classes = {
			'pending': 'warning',
			'in_progress': 'info',
			'completed': 'success',
			'failed': 'danger',
			'rollback': 'warning'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Pre-process before adding new deployment."""
		item.created_by = g.user.username
		item.status = 'pending'
		item.progress_percentage = 0
		
		# Process config
		if hasattr(item, 'config') and isinstance(item.config, str):
			try:
				item.config = json.loads(item.config)
			except json.JSONDecodeError:
				item.config = {}

# =============================================================================
# Export All Views
# =============================================================================

__all__ = [
	# Forms
	'APIConfigForm',
	'PolicyConfigForm', 
	'ConsumerRegistrationForm',
	
	# API Management Views
	'APIManagementView',
	'EndpointManagementView',
	'PolicyManagementView',
	
	# Consumer Management Views
	'ConsumerManagementView',
	'APIKeyManagementView',
	
	# Analytics Views
	'AnalyticsDashboardView',
	'UsageRecordsView',
	
	# Developer Portal Views
	'DeveloperPortalView',
	
	# Deployment Views
	'DeploymentManagementView'
]
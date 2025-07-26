"""
APG API Service Mesh - Flask-AppBuilder Views

Comprehensive web interface views for managing service mesh components including
services, routes, load balancers, policies, and monitoring dashboards.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import render_template, request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget
from wtforms import Form, StringField, IntegerField, BooleanField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length, NumberRange

from .models import (
	SMService, SMEndpoint, SMRoute, SMLoadBalancer, SMPolicy,
	SMMetrics, SMTrace, SMHealthCheck, SMAlert, SMTopology,
	ServiceStatus, EndpointProtocol, LoadBalancerAlgorithm,
	HealthStatus, PolicyType, RouteMatchType
)

# =============================================================================
# Custom Widgets and Forms
# =============================================================================

class ServiceMeshWidget(ListWidget):
	"""Custom widget for service mesh list views."""
	template = 'service_mesh/custom_list.html'

class ServiceRegistrationForm(Form):
	"""Form for service registration."""
	service_name = StringField('Service Name', validators=[DataRequired(), Length(min=1, max=255)])
	service_version = StringField('Service Version', validators=[DataRequired(), Length(min=1, max=50)])
	namespace = StringField('Namespace', default='default')
	description = TextAreaField('Description')
	environment = SelectField('Environment', 
		choices=[('development', 'Development'), ('staging', 'Staging'), ('production', 'Production')],
		default='production'
	)

class EndpointConfigForm(Form):
	"""Form for endpoint configuration."""
	host = StringField('Host', validators=[DataRequired(), Length(min=1, max=255)])
	port = IntegerField('Port', validators=[DataRequired(), NumberRange(min=1, max=65535)])
	protocol = SelectField('Protocol',
		choices=[(p.value, p.value.upper()) for p in EndpointProtocol],
		default=EndpointProtocol.HTTP.value
	)
	path = StringField('Path', default='/')
	weight = IntegerField('Weight', default=100, validators=[NumberRange(min=1)])
	enabled = BooleanField('Enabled', default=True)
	health_check_path = StringField('Health Check Path', default='/health')
	health_check_interval = IntegerField('Health Check Interval (seconds)', default=30)
	tls_enabled = BooleanField('TLS Enabled', default=False)

class RouteConfigForm(Form):
	"""Form for route configuration."""
	route_name = StringField('Route Name', validators=[DataRequired(), Length(min=1, max=255)])
	match_type = SelectField('Match Type',
		choices=[(t.value, t.value.title()) for t in RouteMatchType],
		default=RouteMatchType.PREFIX.value
	)
	match_value = StringField('Match Value', validators=[DataRequired()])
	timeout_ms = IntegerField('Timeout (ms)', default=30000)
	retry_attempts = IntegerField('Retry Attempts', default=3)
	priority = IntegerField('Priority', default=1000)
	enabled = BooleanField('Enabled', default=True)

# =============================================================================
# Service Management Views
# =============================================================================

class SMServiceView(ModelView):
	"""Service management view."""
	datamodel = SQLAInterface(SMService)
	
	# List view configuration
	list_columns = [
		'service_name', 'service_version', 'namespace', 'status', 
		'health_status', 'environment', 'created_at'
	]
	search_columns = ['service_name', 'service_version', 'namespace', 'description']
	order_columns = ['service_name', 'created_at', 'status']
	base_order = ('created_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'service_id', 'service_name', 'service_version', 'namespace',
		'description', 'tags', 'status', 'health_status', 'environment',
		'configuration', 'metadata', 'created_by', 'created_at', 'updated_at'
	]
	
	# Edit view configuration
	edit_columns = [
		'service_name', 'service_version', 'namespace', 'description',
		'tags', 'environment', 'configuration', 'metadata'
	]
	add_columns = edit_columns
	
	# Labels
	label_columns = {
		'service_id': 'Service ID',
		'service_name': 'Service Name',
		'service_version': 'Version',
		'namespace': 'Namespace',
		'health_status': 'Health',
		'created_at': 'Created',
		'updated_at': 'Updated'
	}
	
	# Formatters
	formatters_columns = {
		'status': lambda x: f'<span class="label label-{_get_status_class(x)}">{x}</span>',
		'health_status': lambda x: f'<span class="label label-{_get_health_class(x)}">{x}</span>',
		'tags': lambda x: ', '.join(x) if x else '',
	}
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	@expose('/dashboard/<service_id>')
	@has_access
	def dashboard(self, service_id):
		"""Service-specific dashboard."""
		service = self.datamodel.get(service_id)
		if not service:
			flash('Service not found', 'error')
			return redirect(url_for('SMServiceView.list'))
		
		# Get service metrics and health data
		# This would typically query the metrics and health check tables
		
		return render_template(
			'service_mesh/service_dashboard.html',
			service=service,
			title=f'Service Dashboard - {service.service_name}'
		)

class SMEndpointView(ModelView):
	"""Endpoint management view."""
	datamodel = SQLAInterface(SMEndpoint)
	
	# List view configuration
	list_columns = [
		'service.service_name', 'host', 'port', 'protocol', 'path',
		'weight', 'enabled', 'tls_enabled', 'created_at'
	]
	search_columns = ['host', 'service.service_name']
	order_columns = ['host', 'port', 'created_at']
	
	# Show view configuration
	show_columns = [
		'endpoint_id', 'service.service_name', 'host', 'port', 'protocol',
		'path', 'weight', 'enabled', 'health_check_path', 'health_check_interval',
		'health_check_timeout', 'tls_enabled', 'metadata', 'created_at'
	]
	
	# Edit view configuration
	edit_columns = [
		'service', 'host', 'port', 'protocol', 'path', 'weight', 'enabled',
		'health_check_path', 'health_check_interval', 'health_check_timeout',
		'tls_enabled', 'metadata'
	]
	add_columns = edit_columns
	
	# Labels
	label_columns = {
		'endpoint_id': 'Endpoint ID',
		'service.service_name': 'Service',
		'health_check_path': 'Health Check Path',
		'health_check_interval': 'Check Interval',
		'health_check_timeout': 'Check Timeout',
		'tls_enabled': 'TLS Enabled'
	}
	
	# Related views
	related_views = [SMServiceView]

class SMRouteView(ModelView):
	"""Route management view."""
	datamodel = SQLAInterface(SMRoute)
	
	# List view configuration
	list_columns = [
		'route_name', 'match_type', 'match_value', 'priority',
		'enabled', 'timeout_ms', 'retry_attempts', 'created_at'
	]
	search_columns = ['route_name', 'match_value']
	order_columns = ['route_name', 'priority', 'created_at']
	base_order = ('priority', 'asc')
	
	# Show view configuration
	show_columns = [
		'route_id', 'route_name', 'service.service_name', 'match_type',
		'match_value', 'match_headers', 'match_query', 'destination_services',
		'backup_services', 'timeout_ms', 'retry_attempts', 'retry_timeout_ms',
		'priority', 'enabled', 'request_headers_add', 'response_headers_add',
		'created_at'
	]
	
	# Edit view configuration
	edit_columns = [
		'route_name', 'service', 'match_type', 'match_value', 'match_headers',
		'match_query', 'destination_services', 'backup_services', 'timeout_ms',
		'retry_attempts', 'retry_timeout_ms', 'priority', 'enabled',
		'request_headers_add', 'request_headers_remove',
		'response_headers_add', 'response_headers_remove'
	]
	add_columns = edit_columns
	
	# Labels
	label_columns = {
		'route_id': 'Route ID',
		'route_name': 'Route Name',
		'match_type': 'Match Type',
		'match_value': 'Match Value',
		'match_headers': 'Match Headers',
		'match_query': 'Match Query Parameters',
		'destination_services': 'Destination Services',
		'backup_services': 'Backup Services',
		'timeout_ms': 'Timeout (ms)',
		'retry_attempts': 'Retry Attempts',
		'retry_timeout_ms': 'Retry Timeout (ms)',
		'request_headers_add': 'Add Request Headers',
		'request_headers_remove': 'Remove Request Headers',
		'response_headers_add': 'Add Response Headers',
		'response_headers_remove': 'Remove Response Headers'
	}
	
	# Formatters
	formatters_columns = {
		'enabled': lambda x: '<i class="fa fa-check text-success"></i>' if x else '<i class="fa fa-times text-danger"></i>',
		'destination_services': lambda x: f'{len(x)} services' if x else 'None',
	}

class SMLoadBalancerView(ModelView):
	"""Load balancer management view."""
	datamodel = SQLAInterface(SMLoadBalancer)
	
	# List view configuration
	list_columns = [
		'load_balancer_name', 'service.service_name', 'algorithm',
		'health_check_enabled', 'circuit_breaker_enabled', 'enabled', 'created_at'
	]
	search_columns = ['load_balancer_name', 'service.service_name']
	order_columns = ['load_balancer_name', 'created_at']
	
	# Show view configuration
	show_columns = [
		'load_balancer_id', 'load_balancer_name', 'service.service_name',
		'algorithm', 'session_affinity', 'session_affinity_cookie',
		'health_check_enabled', 'health_check_interval', 'health_check_timeout',
		'healthy_threshold', 'unhealthy_threshold', 'circuit_breaker_enabled',
		'failure_threshold', 'recovery_timeout', 'max_connections',
		'connection_timeout_ms', 'configuration', 'created_at'
	]
	
	# Edit view configuration
	edit_columns = [
		'load_balancer_name', 'service', 'algorithm', 'session_affinity',
		'session_affinity_cookie', 'health_check_enabled', 'health_check_interval',
		'health_check_timeout', 'healthy_threshold', 'unhealthy_threshold',
		'circuit_breaker_enabled', 'failure_threshold', 'recovery_timeout',
		'max_connections', 'connection_timeout_ms', 'configuration', 'enabled'
	]
	add_columns = edit_columns
	
	# Labels
	label_columns = {
		'load_balancer_id': 'Load Balancer ID',
		'load_balancer_name': 'Load Balancer Name',
		'session_affinity': 'Session Affinity',
		'session_affinity_cookie': 'Affinity Cookie',
		'health_check_enabled': 'Health Checks',
		'health_check_interval': 'Check Interval',
		'health_check_timeout': 'Check Timeout',
		'healthy_threshold': 'Healthy Threshold',
		'unhealthy_threshold': 'Unhealthy Threshold',
		'circuit_breaker_enabled': 'Circuit Breaker',
		'failure_threshold': 'Failure Threshold',
		'recovery_timeout': 'Recovery Timeout',
		'max_connections': 'Max Connections',
		'connection_timeout_ms': 'Connection Timeout (ms)'
	}

class SMPolicyView(ModelView):
	"""Policy management view."""
	datamodel = SQLAInterface(SMPolicy)
	
	# List view configuration
	list_columns = [
		'policy_name', 'policy_type', 'route.route_name', 'enabled',
		'priority', 'auth_required', 'created_at'
	]
	search_columns = ['policy_name', 'policy_type', 'description']
	order_columns = ['policy_name', 'policy_type', 'priority', 'created_at']
	base_order = ('priority', 'asc')
	
	# Show view configuration
	show_columns = [
		'policy_id', 'policy_name', 'policy_type', 'route.route_name',
		'configuration', 'enabled', 'priority', 'rate_limit_requests',
		'rate_limit_window_seconds', 'rate_limit_burst', 'auth_required',
		'auth_config', 'description', 'metadata', 'created_at'
	]
	
	# Edit view configuration
	edit_columns = [
		'policy_name', 'policy_type', 'route', 'configuration', 'enabled',
		'priority', 'rate_limit_requests', 'rate_limit_window_seconds',
		'rate_limit_burst', 'auth_required', 'auth_config', 'description', 'metadata'
	]
	add_columns = edit_columns
	
	# Labels
	label_columns = {
		'policy_id': 'Policy ID',
		'policy_name': 'Policy Name',
		'policy_type': 'Policy Type',
		'route.route_name': 'Route',
		'rate_limit_requests': 'Rate Limit (requests)',
		'rate_limit_window_seconds': 'Rate Limit Window (seconds)',
		'rate_limit_burst': 'Rate Limit Burst',
		'auth_required': 'Authentication Required',
		'auth_config': 'Authentication Config'
	}

# =============================================================================
# Monitoring and Analytics Views
# =============================================================================

class SMMetricsView(ModelView):
	"""Metrics monitoring view."""
	datamodel = SQLAInterface(SMMetrics)
	
	# List view configuration (read-only)
	list_columns = [
		'service.service_name', 'metric_name', 'metric_type', 'value',
		'request_count', 'error_count', 'response_time_ms', 'timestamp'
	]
	search_columns = ['service.service_name', 'metric_name', 'metric_type']
	order_columns = ['timestamp', 'service.service_name', 'metric_name']
	base_order = ('timestamp', 'desc')
	
	# Show view configuration
	show_columns = [
		'metric_id', 'service.service_name', 'metric_name', 'metric_type',
		'labels', 'value', 'timestamp', 'request_count', 'error_count',
		'response_time_ms', 'status_code', 'metadata'
	]
	
	# Read-only view
	base_permissions = ['can_list', 'can_show']
	
	# Labels
	label_columns = {
		'metric_id': 'Metric ID',
		'service.service_name': 'Service',
		'metric_name': 'Metric Name',
		'metric_type': 'Metric Type',
		'response_time_ms': 'Response Time (ms)',
		'status_code': 'Status Code'
	}

class SMHealthCheckView(ModelView):
	"""Health check monitoring view."""
	datamodel = SQLAInterface(SMHealthCheck)
	
	# List view configuration (read-only)
	list_columns = [
		'service.service_name', 'check_type', 'status', 'response_time_ms',
		'status_code', 'consecutive_successes', 'consecutive_failures', 'last_check_at'
	]
	search_columns = ['service.service_name', 'status']
	order_columns = ['last_check_at', 'service.service_name', 'status']
	base_order = ('last_check_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'health_check_id', 'service.service_name', 'check_type', 'check_url',
		'check_interval', 'check_timeout', 'status', 'response_time_ms',
		'status_code', 'response_body', 'error_message', 'consecutive_successes',
		'consecutive_failures', 'last_check_at', 'last_success_at', 'last_failure_at'
	]
	
	# Read-only view
	base_permissions = ['can_list', 'can_show']
	
	# Labels
	label_columns = {
		'health_check_id': 'Health Check ID',
		'service.service_name': 'Service',
		'check_type': 'Check Type',
		'check_url': 'Check URL',
		'check_interval': 'Check Interval',
		'check_timeout': 'Check Timeout',
		'response_time_ms': 'Response Time (ms)',
		'status_code': 'Status Code',
		'response_body': 'Response Body',
		'error_message': 'Error Message',
		'consecutive_successes': 'Consecutive Successes',
		'consecutive_failures': 'Consecutive Failures',
		'last_check_at': 'Last Check',
		'last_success_at': 'Last Success',
		'last_failure_at': 'Last Failure'
	}
	
	# Formatters
	formatters_columns = {
		'status': lambda x: f'<span class="label label-{_get_health_class(x)}">{x}</span>',
	}

class SMTopologyView(ModelView):
	"""Service topology view."""
	datamodel = SQLAInterface(SMTopology)
	
	# List view configuration (read-only)
	list_columns = [
		'source_service.service_name', 'target_service.service_name',
		'relationship_type', 'protocol', 'avg_response_time_ms',
		'request_count', 'error_count', 'status'
	]
	search_columns = ['source_service.service_name', 'target_service.service_name', 'relationship_type']
	order_columns = ['source_service.service_name', 'target_service.service_name']
	
	# Show view configuration
	show_columns = [
		'topology_id', 'source_service.service_name', 'target_service.service_name',
		'relationship_type', 'weight', 'protocol', 'port', 'endpoint_path',
		'avg_response_time_ms', 'request_count', 'error_count', 'status',
		'last_communication_at', 'metadata'
	]
	
	# Read-only view
	base_permissions = ['can_list', 'can_show']
	
	# Labels
	label_columns = {
		'topology_id': 'Topology ID',
		'source_service.service_name': 'Source Service',
		'target_service.service_name': 'Target Service',
		'relationship_type': 'Relationship Type',
		'endpoint_path': 'Endpoint Path',
		'avg_response_time_ms': 'Avg Response Time (ms)',
		'request_count': 'Request Count',
		'error_count': 'Error Count',
		'last_communication_at': 'Last Communication'
	}

# =============================================================================
# Dashboard and Custom Views
# =============================================================================

class ServiceMeshDashboardView(BaseView):
	"""Main service mesh dashboard."""
	
	default_view = 'dashboard'
	
	@expose('/dashboard/')
	@has_access
	def dashboard(self):
		"""Main dashboard view."""
		# Get dashboard metrics
		# This would typically aggregate data from the database
		
		dashboard_data = {
			'total_services': 0,
			'healthy_services': 0,
			'total_requests_today': 0,
			'avg_response_time': 0,
			'error_rate': 0,
			'recent_alerts': []
		}
		
		return render_template(
			'service_mesh/dashboard.html',
			dashboard_data=dashboard_data,
			title='Service Mesh Dashboard'
		)
	
	@expose('/topology/')
	@has_access
	def topology(self):
		"""Service topology visualization."""
		return render_template(
			'service_mesh/topology.html',
			title='Service Topology'
		)
	
	@expose('/monitoring/')
	@has_access
	def monitoring(self):
		"""Real-time monitoring dashboard."""
		return render_template(
			'service_mesh/monitoring.html',
			title='Real-time Monitoring'
		)
	
	@expose('/api/dashboard-data/')
	@has_access
	def dashboard_data(self):
		"""API endpoint for dashboard data."""
		# Get real-time dashboard data
		# This would query the database and Redis for current metrics
		
		data = {
			'services': {
				'total': 0,
				'healthy': 0,
				'unhealthy': 0,
				'degraded': 0
			},
			'traffic': {
				'requests_per_second': 0,
				'error_rate': 0,
				'avg_response_time': 0,
				'p95_response_time': 0
			},
			'alerts': {
				'critical': 0,
				'warning': 0,
				'info': 0
			},
			'timestamp': '2025-01-26T20:00:00Z'
		}
		
		return jsonify(data)
	
	@expose('/api/topology-data/')
	@has_access
	def topology_data(self):
		"""API endpoint for topology data."""
		# Get topology graph data
		# This would query the topology table and format for visualization
		
		data = {
			'nodes': [],
			'edges': [],
			'metadata': {
				'total_services': 0,
				'total_connections': 0
			}
		}
		
		return jsonify(data)

class ServiceMeshChartsView(DirectByChartView):
	"""Charts and analytics view."""
	
	datamodel = SQLAInterface(SMMetrics)
	chart_title = 'Service Mesh Metrics'
	
	definitions = [
		{
			'group': 'metric_name',
			'series': ['value'],
		}
	]

# =============================================================================
# Utility Functions
# =============================================================================

def _get_status_class(status):
	"""Get CSS class for service status."""
	status_classes = {
		ServiceStatus.HEALTHY.value: 'success',
		ServiceStatus.UNHEALTHY.value: 'danger',
		ServiceStatus.DEGRADED.value: 'warning',
		ServiceStatus.MAINTENANCE.value: 'info',
		ServiceStatus.REGISTERING.value: 'info',
		ServiceStatus.DEREGISTERING.value: 'warning',
		ServiceStatus.FAILED.value: 'danger'
	}
	return status_classes.get(status, 'default')

def _get_health_class(health_status):
	"""Get CSS class for health status."""
	health_classes = {
		HealthStatus.HEALTHY.value: 'success',
		HealthStatus.UNHEALTHY.value: 'danger',
		HealthStatus.TIMEOUT.value: 'warning',
		HealthStatus.CONNECTION_FAILED.value: 'danger',
		HealthStatus.UNKNOWN.value: 'default'
	}
	return health_classes.get(health_status, 'default')

# Export all view classes
__all__ = [
	'SMServiceView',
	'SMEndpointView', 
	'SMRouteView',
	'SMLoadBalancerView',
	'SMPolicyView',
	'SMMetricsView',
	'SMHealthCheckView',
	'SMTopologyView',
	'ServiceMeshDashboardView',
	'ServiceMeshChartsView'
]
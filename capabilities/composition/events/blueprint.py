"""
APG Event Streaming Bus - Flask-AppBuilder Blueprint

Flask-AppBuilder blueprint registration for event streaming management UI
with comprehensive dashboard, monitoring, and administrative interfaces.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder, SQLA

from .views import (
	EventStreamView, EventView, SubscriptionView, ConsumerGroupView,
	SchemaView, MetricsView, StreamingDashboardView,
	EventSchemaView, StreamAssignmentView, EventProcessingHistoryView,
	StreamProcessorView, EnhancedStreamingDashboardView
)
from .models import (
	ESEvent, ESStream, ESSubscription, ESConsumerGroup, ESSchema, ESMetrics,
	ESEventSchema, ESStreamAssignment, ESEventProcessingHistory, ESStreamProcessor
)

# =============================================================================
# Blueprint Definition
# =============================================================================

def create_event_streaming_blueprint(appbuilder: AppBuilder) -> Blueprint:
	"""
	Create and configure the Event Streaming Bus Flask-AppBuilder blueprint.
	
	Args:
		appbuilder: Flask-AppBuilder instance
		
	Returns:
		Configured blueprint for event streaming management
	"""
	
	# Register model views with AppBuilder
	register_views(appbuilder)
	
	# Register menu structure
	register_menu(appbuilder)
	
	# Return blueprint for further customization if needed
	return Blueprint(
		'event_streaming_bus',
		__name__,
		url_prefix='/esb'
	)

# =============================================================================
# View Registration
# =============================================================================

def register_views(appbuilder: AppBuilder) -> None:
	"""Register all Event Streaming Bus views with AppBuilder."""
	
	# Core streaming management views
	appbuilder.add_view(
		EventStreamView,
		"Event Streams",
		icon="fa-stream",
		category="Event Streaming",
		category_icon="fa-broadcast-tower"
	)
	
	appbuilder.add_view(
		SubscriptionView,
		"Subscriptions",
		icon="fa-rss",
		category="Event Streaming"
	)
	
	appbuilder.add_view(
		ConsumerGroupView,
		"Consumer Groups",
		icon="fa-users",
		category="Event Streaming"
	)
	
	# Enhanced Stream Processing
	appbuilder.add_view(
		StreamProcessorView,
		"Stream Processors",
		icon="fa-cogs",
		category="Stream Processing",
		category_icon="fa-microchip"
	)
	
	appbuilder.add_view(
		StreamAssignmentView,
		"Stream Assignments",
		icon="fa-route",
		category="Stream Processing"
	)
	
	# Event browsing and management
	appbuilder.add_view(
		EventView,
		"Event Browser",
		icon="fa-search",
		category="Event Management",
		category_icon="fa-list-alt"
	)
	
	appbuilder.add_view(
		EventProcessingHistoryView,
		"Processing History",
		icon="fa-history",
		category="Event Management"
	)
	
	# Schema registry - enhanced
	appbuilder.add_view(
		SchemaView,
		"Schema Registry (Legacy)",
		icon="fa-code",
		category="Schema Management",
		category_icon="fa-file-code"
	)
	
	appbuilder.add_view(
		EventSchemaView,
		"Event Schema Registry",
		icon="fa-file-contract",
		category="Schema Management"
	)
	
	# Monitoring and metrics
	appbuilder.add_view(
		MetricsView,
		"Metrics",
		icon="fa-chart-line",
		category="Monitoring",
		category_icon="fa-tachometer-alt"
	)
	
	# Real-time dashboards
	appbuilder.add_view(
		StreamingDashboardView,
		"Dashboard (Classic)",
		icon="fa-dashboard",
		category="Monitoring"
	)
	
	appbuilder.add_view(
		EnhancedStreamingDashboardView,
		"Enhanced Dashboard",
		icon="fa-chart-area",
		category="Monitoring"
	)

# =============================================================================
# Menu Registration
# =============================================================================

def register_menu(appbuilder: AppBuilder) -> None:
	"""Register menu structure for Event Streaming Bus."""
	
	# Enhanced Dashboard as the main entry point
	appbuilder.add_link(
		"Enhanced Dashboard",
		href="/enhanced_dashboard/",
		icon="fa-chart-area",
		category="Event Streaming",
		category_icon="fa-broadcast-tower"
	)
	
	appbuilder.add_link(
		"Classic Dashboard",
		href="/dashboard/",
		icon="fa-dashboard",
		category="Event Streaming"
	)
	
	# Quick access links
	appbuilder.add_separator("Event Streaming")
	
	appbuilder.add_link(
		"Create Stream",
		href="/eventstreamview/add",
		icon="fa-plus",
		category="Event Streaming"
	)
	
	appbuilder.add_link(
		"Create Subscription", 
		href="/subscriptionview/add",
		icon="fa-plus",
		category="Event Streaming"
	)
	
	appbuilder.add_link(
		"Create Stream Processor",
		href="/streamprocessorview/add",
		icon="fa-cogs",
		category="Event Streaming"
	)
	
	# Schema Management shortcuts
	appbuilder.add_separator("Schema Management")
	
	appbuilder.add_link(
		"Register Event Schema",
		href="/eventschemaview/add",
		icon="fa-file-contract",
		category="Schema Management"
	)
	
	appbuilder.add_link(
		"Schema Registry Dashboard",
		href="/enhanced_dashboard/schema_registry",
		icon="fa-database",
		category="Schema Management"
	)
	
	# Stream Processing shortcuts
	appbuilder.add_separator("Stream Processing")
	
	appbuilder.add_link(
		"Processor Metrics",
		href="/enhanced_dashboard/processor_metrics",
		icon="fa-microchip",
		category="Stream Processing"
	)
	
	appbuilder.add_link(
		"Stream Assignments",
		href="/streamassignmentview/list",
		icon="fa-route",
		category="Stream Processing"
	)
	
	# Monitoring shortcuts
	appbuilder.add_separator("Monitoring")
	
	appbuilder.add_link(
		"Stream Topology",
		href="/dashboard/stream_topology",
		icon="fa-project-diagram",
		category="Monitoring"
	)
	
	appbuilder.add_link(
		"Live Metrics",
		href="/dashboard/api/live_metrics",
		icon="fa-chart-line",
		category="Monitoring"
	)
	
	appbuilder.add_link(
		"Event Browser",
		href="/dashboard/event_browser",
		icon="fa-search",
		category="Monitoring"
	)
	
	appbuilder.add_link(
		"Processing History",
		href="/eventprocessinghistoryview/list",
		icon="fa-history",
		category="Monitoring"
	)

# =============================================================================
# Security Configuration
# =============================================================================

def configure_security(appbuilder: AppBuilder) -> None:
	"""Configure security settings for Event Streaming Bus views."""
	
	# Define custom roles if needed
	
	# Event Stream Manager role - enhanced
	stream_manager_permissions = [
		'can_list_on_EventStreamView',
		'can_show_on_EventStreamView', 
		'can_add_on_EventStreamView',
		'can_edit_on_EventStreamView',
		'can_delete_on_EventStreamView',
		'can_pause_stream_on_EventStreamView',
		'can_resume_stream_on_EventStreamView',
		'can_stream_metrics_on_EventStreamView',
		# Stream processor permissions
		'can_list_on_StreamProcessorView',
		'can_show_on_StreamProcessorView',
		'can_add_on_StreamProcessorView',
		'can_edit_on_StreamProcessorView',
		'can_delete_on_StreamProcessorView',
		'can_start_processor_on_StreamProcessorView',
		'can_stop_processor_on_StreamProcessorView',
		# Stream assignment permissions
		'can_list_on_StreamAssignmentView',
		'can_show_on_StreamAssignmentView',
		'can_add_on_StreamAssignmentView',
		'can_edit_on_StreamAssignmentView',
		'can_delete_on_StreamAssignmentView'
	]
	
	# Subscription Manager role
	subscription_manager_permissions = [
		'can_list_on_SubscriptionView',
		'can_show_on_SubscriptionView',
		'can_add_on_SubscriptionView', 
		'can_edit_on_SubscriptionView',
		'can_delete_on_SubscriptionView',
		'can_subscription_lag_on_SubscriptionView'
	]
	
	# Event Reader role (read-only access) - enhanced
	event_reader_permissions = [
		'can_list_on_EventView',
		'can_show_on_EventView',
		'can_event_trace_on_EventView',
		'can_list_on_MetricsView',
		'can_show_on_MetricsView',
		'can_index_on_StreamingDashboardView',
		'can_live_metrics_on_StreamingDashboardView',
		'can_stream_topology_on_StreamingDashboardView',
		'can_event_browser_on_StreamingDashboardView',
		# Enhanced dashboard permissions
		'can_index_on_EnhancedStreamingDashboardView',
		'can_processor_metrics_on_EnhancedStreamingDashboardView',
		'can_schema_registry_on_EnhancedStreamingDashboardView',
		# Processing history permissions
		'can_list_on_EventProcessingHistoryView',
		'can_show_on_EventProcessingHistoryView'
	]
	
	# Schema Manager role - enhanced
	schema_manager_permissions = [
		'can_list_on_SchemaView',
		'can_show_on_SchemaView',
		'can_add_on_SchemaView',
		'can_edit_on_SchemaView',
		'can_delete_on_SchemaView',
		# Enhanced schema permissions
		'can_list_on_EventSchemaView',
		'can_show_on_EventSchemaView',
		'can_add_on_EventSchemaView',
		'can_edit_on_EventSchemaView',
		'can_delete_on_EventSchemaView'
	]
	
	# Stream Processing Administrator role - new
	stream_processing_admin_permissions = [
		'can_list_on_StreamProcessorView',
		'can_show_on_StreamProcessorView',
		'can_add_on_StreamProcessorView',
		'can_edit_on_StreamProcessorView',
		'can_delete_on_StreamProcessorView',
		'can_start_processor_on_StreamProcessorView',
		'can_stop_processor_on_StreamProcessorView',
		'can_list_on_StreamAssignmentView',
		'can_show_on_StreamAssignmentView',
		'can_add_on_StreamAssignmentView',
		'can_edit_on_StreamAssignmentView',
		'can_delete_on_StreamAssignmentView',
		'can_list_on_EventProcessingHistoryView',
		'can_show_on_EventProcessingHistoryView',
		'can_delete_on_EventProcessingHistoryView'
	]
	
	# Create roles if they don't exist
	try:
		# Stream Manager role
		stream_manager_role = appbuilder.sm.find_role('Event Stream Manager')
		if not stream_manager_role:
			stream_manager_role = appbuilder.sm.add_role('Event Stream Manager')
			for perm in stream_manager_permissions:
				permission = appbuilder.sm.find_permission_on_view(
					perm.split('_on_')[0], perm.split('_on_')[1]
				)
				if permission:
					appbuilder.sm.add_permission_role(stream_manager_role, permission)
		
		# Subscription Manager role
		subscription_manager_role = appbuilder.sm.find_role('Subscription Manager')
		if not subscription_manager_role:
			subscription_manager_role = appbuilder.sm.add_role('Subscription Manager')
			for perm in subscription_manager_permissions:
				permission = appbuilder.sm.find_permission_on_view(
					perm.split('_on_')[0], perm.split('_on_')[1]
				)
				if permission:
					appbuilder.sm.add_permission_role(subscription_manager_role, permission)
		
		# Event Reader role
		event_reader_role = appbuilder.sm.find_role('Event Reader')
		if not event_reader_role:
			event_reader_role = appbuilder.sm.add_role('Event Reader')
			for perm in event_reader_permissions:
				permission = appbuilder.sm.find_permission_on_view(
					perm.split('_on_')[0], perm.split('_on_')[1]
				)
				if permission:
					appbuilder.sm.add_permission_role(event_reader_role, permission)
		
		# Schema Manager role
		schema_manager_role = appbuilder.sm.find_role('Schema Manager')
		if not schema_manager_role:
			schema_manager_role = appbuilder.sm.add_role('Schema Manager')
			for perm in schema_manager_permissions:
				permission = appbuilder.sm.find_permission_on_view(
					perm.split('_on_')[0], perm.split('_on_')[1]
				)
				if permission:
					appbuilder.sm.add_permission_role(schema_manager_role, permission)
		
		# Stream Processing Administrator role
		stream_processing_admin_role = appbuilder.sm.find_role('Stream Processing Administrator')
		if not stream_processing_admin_role:
			stream_processing_admin_role = appbuilder.sm.add_role('Stream Processing Administrator')
			for perm in stream_processing_admin_permissions:
				permission = appbuilder.sm.find_permission_on_view(
					perm.split('_on_')[0], perm.split('_on_')[1]
				)
				if permission:
					appbuilder.sm.add_permission_role(stream_processing_admin_role, permission)
		
	except Exception as e:
		# Log error but don't fail - roles can be created manually
		print(f"Warning: Could not create Event Streaming roles: {e}")

# =============================================================================
# Template Configuration
# =============================================================================

def register_templates(appbuilder: AppBuilder) -> None:
	"""Register custom templates for Event Streaming Bus views."""
	
	# Custom templates would be registered here
	# These would be stored in templates/event_streaming_bus/ directory
	
	templates = [
		'streaming_dashboard.html',
		'enhanced_streaming_dashboard.html',
		'stream_metrics.html', 
		'subscription_lag.html',
		'event_trace.html',
		'stream_topology.html',
		'event_browser.html',
		'processor_metrics.html',
		'schema_registry.html',
		'stream_assignment_detail.html',
		'processing_history_chart.html'
	]
	
	# Templates would be automatically found by Flask-AppBuilder
	# if placed in the correct directory structure

# =============================================================================
# Blueprint Initialization Function
# =============================================================================

def init_event_streaming_blueprint(app, appbuilder: AppBuilder):
	"""
	Initialize the Event Streaming Bus blueprint with an existing Flask app.
	
	Args:
		app: Flask application instance
		appbuilder: Flask-AppBuilder instance
	"""
	
	# Create and register the blueprint
	blueprint = create_event_streaming_blueprint(appbuilder)
	app.register_blueprint(blueprint)
	
	# Configure security
	configure_security(appbuilder)
	
	# Register templates
	register_templates(appbuilder)
	
	return blueprint

# =============================================================================
# APG Integration Functions
# =============================================================================

def register_apg_integration(apg_registry):
	"""Register Event Streaming Bus capability with APG platform."""
	
	capability_info = {
		'capability_id': 'event_streaming_bus',
		'capability_name': 'Event Streaming Bus',
		'capability_version': '1.0.0',
		'capability_description': 'Enterprise-grade event streaming platform with Apache Kafka',
		'capability_category': 'composition_orchestration',
		'author': 'Nyimbi Odero',
		'company': 'Datacraft',
		'requires_dependencies': [
			'apache_kafka',
			'redis',
			'postgresql'
		],
		'provides_services': [
			'event_streaming',
			'event_sourcing',
			'stream_processing',
			'schema_registry',
			'consumer_management'
		],
		'api_endpoints': {
			'base_url': '/api/v1/event-streaming',
			'health_check': '/health',
			'metrics': '/metrics',
			'documentation': '/docs'
		},
		'ui_routes': {
			'dashboard': '/enhanced_dashboard/',
			'streams': '/eventstreamview/list',
			'processors': '/streamprocessorview/list',
			'schemas': '/eventschemaview/list'
		},
		'configuration_schema': {
			'kafka_brokers': {
				'type': 'string',
				'required': True,
				'description': 'Kafka broker connection string'
			},
			'redis_url': {
				'type': 'string',
				'required': True,
				'description': 'Redis connection URL'
			},
			'default_partitions': {
				'type': 'integer',
				'default': 3,
				'description': 'Default number of partitions for new streams'
			},
			'default_replication_factor': {
				'type': 'integer',
				'default': 2,
				'description': 'Default replication factor for new streams'
			}
		},
		'monitoring': {
			'prometheus_metrics': True,
			'health_checks': True,
			'distributed_tracing': True,
			'logging': True
		},
		'security': {
			'authentication_required': True,
			'authorization_enabled': True,
			'tenant_isolation': True,
			'encryption_at_rest': True,
			'encryption_in_transit': True
		}
	}
	
	# Register with APG platform
	if hasattr(apg_registry, 'register_capability'):
		apg_registry.register_capability(capability_info)
	
	return capability_info

def get_capability_metadata():
	"""Get capability metadata for APG discovery."""
	return {
		'models': [
			'ESEvent', 'ESStream', 'ESSubscription', 'ESConsumerGroup', 
			'ESSchema', 'ESMetrics', 'ESEventSchema', 'ESStreamAssignment',
			'ESEventProcessingHistory', 'ESStreamProcessor'
		],
		'views': [
			'EventStreamView', 'EventView', 'SubscriptionView', 'ConsumerGroupView',
			'SchemaView', 'MetricsView', 'StreamingDashboardView', 'EventSchemaView',
			'StreamAssignmentView', 'EventProcessingHistoryView', 'StreamProcessorView',
			'EnhancedStreamingDashboardView'
		],
		'services': [
			'EventStreamingService', 'EventPublishingService', 'EventConsumptionService',
			'StreamProcessingService', 'EventSourcingService', 'SchemaRegistryService',
			'StreamManagementService', 'ConsumerManagementService'
		],
		'api_routes': [
			'/api/v1/events', '/api/v1/streams', '/api/v1/subscriptions',
			'/api/v1/schemas', '/api/v1/event-sourcing', '/api/v1/stream-processors',
			'/api/v1/consumer-groups'
		],
		'websocket_endpoints': [
			'/ws/events/{stream_name}', '/ws/subscriptions/{subscription_id}',
			'/ws/monitoring'
		],
		'database_tables': [
			'es_events', 'es_streams', 'es_subscriptions', 'es_consumer_groups',
			'es_schemas', 'es_metrics', 'es_event_schemas', 'es_stream_assignments',
			'es_event_processing_history', 'es_stream_processors', 'es_audit_logs'
		]
	}

# Export main functions
__all__ = [
	'create_event_streaming_blueprint',
	'init_event_streaming_blueprint',
	'register_views',
	'register_menu', 
	'configure_security',
	'register_apg_integration',
	'get_capability_metadata'
]
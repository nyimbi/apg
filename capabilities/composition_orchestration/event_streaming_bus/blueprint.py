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
	SchemaView, MetricsView, StreamingDashboardView
)
from .models import ESEvent, ESStream, ESSubscription, ESConsumerGroup, ESSchema, ESMetrics

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
	
	# Event browsing and management
	appbuilder.add_view(
		EventView,
		"Event Browser",
		icon="fa-search",
		category="Event Management",
		category_icon="fa-list-alt"
	)
	
	# Schema registry
	appbuilder.add_view(
		SchemaView,
		"Schema Registry",
		icon="fa-code",
		category="Event Management"
	)
	
	# Monitoring and metrics
	appbuilder.add_view(
		MetricsView,
		"Metrics",
		icon="fa-chart-line",
		category="Monitoring",
		category_icon="fa-tachometer-alt"
	)
	
	# Real-time dashboard
	appbuilder.add_view(
		StreamingDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Monitoring"
	)

# =============================================================================
# Menu Registration
# =============================================================================

def register_menu(appbuilder: AppBuilder) -> None:
	"""Register menu structure for Event Streaming Bus."""
	
	# Dashboard as the main entry point
	appbuilder.add_link(
		"Streaming Dashboard",
		href="/dashboard/",
		icon="fa-dashboard",
		category="Event Streaming",
		category_icon="fa-broadcast-tower"
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

# =============================================================================
# Security Configuration
# =============================================================================

def configure_security(appbuilder: AppBuilder) -> None:
	"""Configure security settings for Event Streaming Bus views."""
	
	# Define custom roles if needed
	
	# Event Stream Manager role
	stream_manager_permissions = [
		'can_list_on_EventStreamView',
		'can_show_on_EventStreamView', 
		'can_add_on_EventStreamView',
		'can_edit_on_EventStreamView',
		'can_delete_on_EventStreamView',
		'can_pause_stream_on_EventStreamView',
		'can_resume_stream_on_EventStreamView',
		'can_stream_metrics_on_EventStreamView'
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
	
	# Event Reader role (read-only access)
	event_reader_permissions = [
		'can_list_on_EventView',
		'can_show_on_EventView',
		'can_event_trace_on_EventView',
		'can_list_on_MetricsView',
		'can_show_on_MetricsView',
		'can_index_on_StreamingDashboardView',
		'can_live_metrics_on_StreamingDashboardView',
		'can_stream_topology_on_StreamingDashboardView',
		'can_event_browser_on_StreamingDashboardView'
	]
	
	# Schema Manager role
	schema_manager_permissions = [
		'can_list_on_SchemaView',
		'can_show_on_SchemaView',
		'can_add_on_SchemaView',
		'can_edit_on_SchemaView',
		'can_delete_on_SchemaView'
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
		'stream_metrics.html', 
		'subscription_lag.html',
		'event_trace.html',
		'stream_topology.html',
		'event_browser.html'
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

# Export main functions
__all__ = [
	'create_event_streaming_blueprint',
	'init_event_streaming_blueprint',
	'register_views',
	'register_menu', 
	'configure_security'
]
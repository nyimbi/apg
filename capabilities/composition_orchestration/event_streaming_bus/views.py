"""
APG Event Streaming Bus - Flask-AppBuilder Views

Comprehensive UI views for event streaming management, monitoring, and 
real-time dashboard with WebSocket support for live updates.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from flask import flash, request, jsonify, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import ChartView
from flask_appbuilder.widgets import ListWidget, FormWidget
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, BS3TextAreaFieldWidget
from wtforms import TextAreaField, SelectField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Length, NumberRange

from .models import (
	ESEvent, ESStream, ESSubscription, ESConsumerGroup, ESSchema, 
	ESMetrics, ESAuditLog, ESEventSchema, ESStreamAssignment, 
	ESEventProcessingHistory, ESStreamProcessor,
	EventStatus, EventPriority, StreamStatus, SubscriptionStatus, ConsumerStatus, ProcessorType,
	EventType, DeliveryMode, CompressionType, SerializationFormat
)
from .service import EventStreamingService, EventPublishingService, EventConsumptionService

# =============================================================================
# Custom Widgets and Forms
# =============================================================================

class JSONTextAreaWidget(BS3TextAreaFieldWidget):
	"""Custom widget for JSON field editing with syntax highlighting."""
	
	def __call__(self, field, **kwargs):
		kwargs.setdefault('rows', 10)
		kwargs.setdefault('class_', 'form-control json-editor')
		kwargs.setdefault('data-mode', 'json')
		return super(JSONTextAreaWidget, self).__call__(field, **kwargs)

class StreamConfigWidget(BS3TextAreaFieldWidget):
	"""Widget for stream configuration with validation."""
	
	def __call__(self, field, **kwargs):
		kwargs.setdefault('rows', 15)
		kwargs.setdefault('class_', 'form-control config-editor')
		kwargs.setdefault('placeholder', 'Enter JSON configuration...')
		return super(StreamConfigWidget, self).__call__(field, **kwargs)

# =============================================================================
# Event Stream Management Views
# =============================================================================

class EventStreamView(ModelView):
	"""Management interface for event streams."""
	
	datamodel = SQLAInterface(ESStream)
	list_title = "Event Streams"
	show_title = "Stream Details"
	add_title = "Create Event Stream"
	edit_title = "Edit Event Stream"
	
	# List view configuration
	list_columns = [
		'stream_name', 'topic_name', 'source_capability', 'status',
		'partitions', 'replication_factor', 'event_category', 'created_at'
	]
	
	search_columns = [
		'stream_name', 'topic_name', 'source_capability', 'event_category', 'status'
	]
	
	order_columns = ['stream_name', 'created_at', 'status']
	base_order = ('created_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'stream_id', 'stream_name', 'stream_description', 'topic_name',
		'partitions', 'replication_factor', 'retention_time_ms', 'retention_size_bytes',
		'compression_type', 'default_serialization', 'event_category',
		'source_capability', 'status', 'tenant_id', 'created_at', 'updated_at', 'created_by'
	]
	
	# Form configuration
	add_columns = [
		'stream_name', 'stream_description', 'topic_name', 'partitions',
		'replication_factor', 'retention_time_ms', 'retention_size_bytes',
		'cleanup_policy', 'compression_type', 'default_serialization',
		'event_category', 'source_capability', 'config_settings'
	]
	
	edit_columns = [
		'stream_description', 'partitions', 'retention_time_ms', 'retention_size_bytes',
		'cleanup_policy', 'compression_type', 'status', 'config_settings'
	]
	
	# Field customization
	add_form_extra_fields = {
		'config_settings': TextAreaField(
			'Configuration Settings',
			widget=JSONTextAreaWidget(),
			description='JSON configuration for advanced stream settings'
		)
	}
	
	edit_form_extra_fields = add_form_extra_fields
	
	# Labels and descriptions
	label_columns = {
		'stream_id': 'Stream ID',
		'stream_name': 'Stream Name',
		'stream_description': 'Description',
		'topic_name': 'Kafka Topic',
		'partitions': 'Partitions',
		'replication_factor': 'Replication',
		'retention_time_ms': 'Retention Time (ms)',
		'retention_size_bytes': 'Retention Size (bytes)',
		'cleanup_policy': 'Cleanup Policy',
		'compression_type': 'Compression',
		'default_serialization': 'Serialization',
		'event_category': 'Event Category',
		'source_capability': 'Source Capability',
		'config_settings': 'Configuration',
		'status': 'Status',
		'tenant_id': 'Tenant',
		'created_at': 'Created',
		'updated_at': 'Updated',
		'created_by': 'Created By'
	}
	
	description_columns = {
		'stream_name': 'Unique name for the event stream',
		'topic_name': 'Underlying Kafka topic name',
		'partitions': 'Number of partitions for parallel processing',
		'replication_factor': 'Number of replicas for fault tolerance',
		'retention_time_ms': 'How long to retain events (milliseconds)',
		'retention_size_bytes': 'Maximum size before cleanup (bytes)',
		'config_settings': 'Advanced JSON configuration settings'
	}
	
	# Formatters
	formatters_columns = {
		'retention_time_ms': lambda x: f"{x // (1000 * 60 * 60 * 24)} days" if x else "N/A",
		'retention_size_bytes': lambda x: f"{x // (1024**3):.1f} GB" if x else "N/A",
		'config_settings': lambda x: json.dumps(x, indent=2)[:100] + "..." if x else "{}"
	}
	
	@expose('/stream_metrics/<stream_id>')
	@has_access
	def stream_metrics(self, stream_id):
		"""Display metrics for a specific stream."""
		stream = self.datamodel.get(stream_id)
		if not stream:
			flash('Stream not found', 'error')
			return self.list()
		
		# Get recent metrics
		metrics = ESMetrics.query.filter_by(stream_id=stream_id)\
			.filter(ESMetrics.time_bucket >= datetime.utcnow() - timedelta(hours=24))\
			.order_by(ESMetrics.time_bucket.desc()).limit(100).all()
		
		return self.render_template(
			'stream_metrics.html',
			stream=stream,
			metrics=metrics,
			title=f"Metrics - {stream.stream_name}"
		)
	
	@expose('/pause_stream/<stream_id>')
	@has_access
	def pause_stream(self, stream_id):
		"""Pause event stream."""
		try:
			stream = self.datamodel.get(stream_id)
			if stream:
				stream.status = StreamStatus.PAUSED.value
				self.datamodel.edit(stream)
				flash(f'Stream {stream.stream_name} paused successfully', 'success')
			else:
				flash('Stream not found', 'error')
		except Exception as e:
			flash(f'Error pausing stream: {str(e)}', 'error')
		
		return self.list()
	
	@expose('/resume_stream/<stream_id>')
	@has_access  
	def resume_stream(self, stream_id):
		"""Resume paused event stream."""
		try:
			stream = self.datamodel.get(stream_id)
			if stream:
				stream.status = StreamStatus.ACTIVE.value
				self.datamodel.edit(stream)
				flash(f'Stream {stream.stream_name} resumed successfully', 'success')
			else:
				flash('Stream not found', 'error')
		except Exception as e:
			flash(f'Error resuming stream: {str(e)}', 'error')
		
		return self.list()

# =============================================================================
# Event Browser and Management
# =============================================================================

class EventView(ModelView):
	"""Browse and manage individual events."""
	
	datamodel = SQLAInterface(ESEvent)
	list_title = "Event Browser"
	show_title = "Event Details"
	
	# List view configuration
	list_columns = [
		'event_id', 'event_type', 'source_capability', 'aggregate_id',
		'status', 'stream_id', 'timestamp', 'created_by'
	]
	
	search_columns = [
		'event_id', 'event_type', 'source_capability', 'aggregate_id',
		'aggregate_type', 'correlation_id', 'causation_id', 'stream_id'
	]
	
	order_columns = ['timestamp', 'event_type', 'status']
	base_order = ('timestamp', 'desc')
	
	# Show view configuration  
	show_columns = [
		'event_id', 'event_type', 'event_version', 'source_capability',
		'aggregate_id', 'aggregate_type', 'sequence_number', 'timestamp',
		'correlation_id', 'causation_id', 'tenant_id', 'user_id',
		'payload', 'metadata', 'schema_id', 'schema_version',
		'serialization_format', 'status', 'retry_count', 'max_retries',
		'stream_id', 'partition_key', 'offset_position', 'created_at', 'created_by'
	]
	
	# No add/edit - events are immutable
	add_columns = []
	edit_columns = []
	can_create = False
	can_edit = False
	can_delete = False
	
	# Labels
	label_columns = {
		'event_id': 'Event ID',
		'event_type': 'Type',
		'event_version': 'Version',
		'source_capability': 'Source',
		'aggregate_id': 'Aggregate ID',
		'aggregate_type': 'Aggregate Type',
		'sequence_number': 'Sequence',
		'timestamp': 'Timestamp',
		'correlation_id': 'Correlation ID',
		'causation_id': 'Causation ID',
		'tenant_id': 'Tenant',
		'user_id': 'User',
		'payload': 'Payload',
		'metadata': 'Metadata',
		'schema_id': 'Schema ID',
		'schema_version': 'Schema Version',
		'serialization_format': 'Format',
		'status': 'Status',
		'retry_count': 'Retries',
		'max_retries': 'Max Retries',
		'stream_id': 'Stream',
		'partition_key': 'Partition Key',
		'offset_position': 'Offset',
		'created_at': 'Created',
		'created_by': 'Created By'
	}
	
	# Formatters
	formatters_columns = {
		'payload': lambda x: json.dumps(x, indent=2) if x else "{}",
		'metadata': lambda x: json.dumps(x, indent=2) if x else "{}",
		'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S UTC') if x else "",
		'created_at': lambda x: x.strftime('%Y-%m-%d %H:%M:%S UTC') if x else ""
	}
	
	@expose('/event_trace/<event_id>')
	@has_access
	def event_trace(self, event_id):
		"""Trace event correlation and causation chain."""
		event = self.datamodel.get(event_id)
		if not event:
			flash('Event not found', 'error')
			return self.list()
		
		# Find related events
		related_events = []
		
		# Events with same correlation_id
		if event.correlation_id:
			correlated = ESEvent.query.filter_by(correlation_id=event.correlation_id)\
				.filter(ESEvent.event_id != event_id).all()
			related_events.extend(correlated)
		
		# Events caused by this event
		caused_events = ESEvent.query.filter_by(causation_id=event_id).all()
		related_events.extend(caused_events)
		
		# Event that caused this event
		if event.causation_id:
			causing_event = ESEvent.query.filter_by(event_id=event.causation_id).first()
			if causing_event:
				related_events.append(causing_event)
		
		return self.render_template(
			'event_trace.html',
			event=event,
			related_events=related_events,
			title=f"Event Trace - {event.event_id}"
		)

# =============================================================================
# Subscription Management
# =============================================================================

class SubscriptionView(ModelView):
	"""Manage event subscriptions and consumers."""
	
	datamodel = SQLAInterface(ESSubscription)
	list_title = "Event Subscriptions"
	show_title = "Subscription Details"
	add_title = "Create Subscription"
	edit_title = "Edit Subscription"
	
	# List view configuration
	list_columns = [
		'subscription_name', 'stream_id', 'consumer_group_id',
		'consumer_name', 'status', 'delivery_mode', 'last_consumed_at', 'created_at'
	]
	
	search_columns = [
		'subscription_name', 'stream_id', 'consumer_group_id',
		'consumer_name', 'status', 'webhook_url'
	]
	
	order_columns = ['subscription_name', 'created_at', 'last_consumed_at']
	base_order = ('created_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'subscription_id', 'subscription_name', 'subscription_description',
		'stream_id', 'consumer_group_id', 'consumer_name', 'event_type_patterns',
		'filter_criteria', 'delivery_mode', 'batch_size', 'max_wait_time_ms',
		'start_position', 'specific_offset', 'retry_policy', 'dead_letter_enabled',
		'dead_letter_topic', 'webhook_url', 'webhook_headers', 'webhook_timeout_ms',
		'status', 'last_consumed_offset', 'last_consumed_at', 'tenant_id',
		'created_at', 'updated_at', 'created_by'
	]
	
	# Form configuration
	add_columns = [
		'subscription_name', 'subscription_description', 'stream_id',
		'consumer_group_id', 'consumer_name', 'event_type_patterns',
		'filter_criteria', 'delivery_mode', 'batch_size', 'max_wait_time_ms',
		'start_position', 'specific_offset', 'retry_policy', 'dead_letter_enabled',
		'dead_letter_topic', 'webhook_url', 'webhook_headers', 'webhook_timeout_ms'
	]
	
	edit_columns = [
		'subscription_description', 'event_type_patterns', 'filter_criteria',
		'delivery_mode', 'batch_size', 'max_wait_time_ms', 'retry_policy',
		'dead_letter_enabled', 'dead_letter_topic', 'webhook_url',
		'webhook_headers', 'webhook_timeout_ms', 'status'
	]
	
	# Field customization
	add_form_extra_fields = {
		'event_type_patterns': TextAreaField(
			'Event Type Patterns',
			widget=JSONTextAreaWidget(),
			description='JSON array of event type patterns (e.g., ["user.*", "order.created"])'
		),
		'filter_criteria': TextAreaField(
			'Filter Criteria',
			widget=JSONTextAreaWidget(),
			description='JSON object with additional filter conditions'
		),
		'retry_policy': TextAreaField(
			'Retry Policy',
			widget=JSONTextAreaWidget(),
			description='JSON object defining retry behavior'
		),
		'webhook_headers': TextAreaField(
			'Webhook Headers',
			widget=JSONTextAreaWidget(),
			description='JSON object with HTTP headers for webhook delivery'
		)
	}
	
	edit_form_extra_fields = add_form_extra_fields
	
	# Labels
	label_columns = {
		'subscription_id': 'Subscription ID',
		'subscription_name': 'Name',
		'subscription_description': 'Description',
		'stream_id': 'Stream',
		'consumer_group_id': 'Consumer Group',
		'consumer_name': 'Consumer Name',
		'event_type_patterns': 'Event Patterns',
		'filter_criteria': 'Filters',
		'delivery_mode': 'Delivery Mode',
		'batch_size': 'Batch Size',
		'max_wait_time_ms': 'Max Wait (ms)',
		'start_position': 'Start Position',
		'specific_offset': 'Specific Offset',
		'retry_policy': 'Retry Policy',
		'dead_letter_enabled': 'Dead Letter Queue',
		'dead_letter_topic': 'DLQ Topic',
		'webhook_url': 'Webhook URL',
		'webhook_headers': 'Webhook Headers',
		'webhook_timeout_ms': 'Webhook Timeout',
		'status': 'Status',
		'last_consumed_offset': 'Last Offset',
		'last_consumed_at': 'Last Consumed',
		'tenant_id': 'Tenant',
		'created_at': 'Created',
		'updated_at': 'Updated',
		'created_by': 'Created By'
	}
	
	# Formatters
	formatters_columns = {
		'event_type_patterns': lambda x: json.dumps(x) if x else "[]",
		'filter_criteria': lambda x: json.dumps(x, indent=2) if x else "{}",
		'retry_policy': lambda x: json.dumps(x, indent=2) if x else "{}",
		'webhook_headers': lambda x: json.dumps(x, indent=2) if x else "{}",
		'last_consumed_at': lambda x: x.strftime('%Y-%m-%d %H:%M:%S UTC') if x else "Never"
	}
	
	@expose('/subscription_lag/<subscription_id>')
	@has_access
	def subscription_lag(self, subscription_id):
		"""Show consumer lag for subscription."""
		subscription = self.datamodel.get(subscription_id)
		if not subscription:
			flash('Subscription not found', 'error')
			return self.list()
		
		# Calculate lag (simplified - would need Kafka admin client in production)
		lag_data = {
			'subscription': subscription,
			'consumer_lag': 0,  # Would calculate from Kafka
			'last_processed': subscription.last_consumed_offset or 0,
			'latest_offset': 0,  # Would get from Kafka
			'processing_rate': 0  # Events per second
		}
		
		return self.render_template(
			'subscription_lag.html',
			**lag_data,
			title=f"Consumer Lag - {subscription.subscription_name}"
		)

# =============================================================================
# Consumer Group Management
# =============================================================================

class ConsumerGroupView(ModelView):
	"""Manage consumer groups and their state."""
	
	datamodel = SQLAInterface(ESConsumerGroup)
	list_title = "Consumer Groups"
	show_title = "Consumer Group Details"
	add_title = "Create Consumer Group"
	edit_title = "Edit Consumer Group"
	
	# List view configuration
	list_columns = [
		'group_name', 'active_consumers', 'total_lag',
		'partition_assignment_strategy', 'tenant_id', 'created_at'
	]
	
	search_columns = ['group_name', 'group_id', 'tenant_id']
	order_columns = ['group_name', 'created_at', 'active_consumers']
	base_order = ('created_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'group_id', 'group_name', 'group_description', 'session_timeout_ms',
		'heartbeat_interval_ms', 'max_poll_interval_ms', 'partition_assignment_strategy',
		'rebalance_timeout_ms', 'active_consumers', 'total_lag', 'tenant_id',
		'created_at', 'updated_at', 'created_by'
	]
	
	# Form configuration
	add_columns = [
		'group_id', 'group_name', 'group_description', 'session_timeout_ms',
		'heartbeat_interval_ms', 'max_poll_interval_ms', 'partition_assignment_strategy',
		'rebalance_timeout_ms'
	]
	
	edit_columns = [
		'group_description', 'session_timeout_ms', 'heartbeat_interval_ms',
		'max_poll_interval_ms', 'partition_assignment_strategy', 'rebalance_timeout_ms'
	]
	
	# Labels
	label_columns = {
		'group_id': 'Group ID',
		'group_name': 'Group Name',
		'group_description': 'Description',
		'session_timeout_ms': 'Session Timeout (ms)',
		'heartbeat_interval_ms': 'Heartbeat Interval (ms)',
		'max_poll_interval_ms': 'Max Poll Interval (ms)',
		'partition_assignment_strategy': 'Assignment Strategy',
		'rebalance_timeout_ms': 'Rebalance Timeout (ms)',
		'active_consumers': 'Active Consumers',
		'total_lag': 'Total Lag',
		'tenant_id': 'Tenant',
		'created_at': 'Created',
		'updated_at': 'Updated',
		'created_by': 'Created By'
	}

# =============================================================================
# Schema Registry Management  
# =============================================================================

class SchemaView(ModelView):
	"""Manage event schemas and validation rules."""
	
	datamodel = SQLAInterface(ESSchema)
	list_title = "Schema Registry"
	show_title = "Schema Details"
	add_title = "Register Schema"
	edit_title = "Edit Schema"
	
	# List view configuration
	list_columns = [
		'schema_name', 'schema_version', 'event_type', 'schema_format',
		'compatibility_level', 'is_active', 'created_at'
	]
	
	search_columns = [
		'schema_name', 'event_type', 'schema_format', 'compatibility_level'
	]
	
	order_columns = ['schema_name', 'schema_version', 'created_at']
	base_order = ('created_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'schema_id', 'schema_name', 'schema_version', 'schema_definition',
		'schema_format', 'event_type', 'compatibility_level', 'is_active',
		'tenant_id', 'created_at', 'updated_at', 'created_by'
	]
	
	# Form configuration
	add_columns = [
		'schema_name', 'schema_version', 'schema_definition', 'schema_format',
		'event_type', 'compatibility_level', 'is_active'
	]
	
	edit_columns = [
		'schema_definition', 'compatibility_level', 'is_active'
	]
	
	# Field customization
	add_form_extra_fields = {
		'schema_definition': TextAreaField(
			'Schema Definition',
			widget=JSONTextAreaWidget(),
			description='JSON Schema definition for event validation',
			validators=[DataRequired()]
		)
	}
	
	edit_form_extra_fields = add_form_extra_fields
	
	# Labels
	label_columns = {
		'schema_id': 'Schema ID',
		'schema_name': 'Schema Name',
		'schema_version': 'Version',
		'schema_definition': 'Definition',
		'schema_format': 'Format',
		'event_type': 'Event Type',
		'compatibility_level': 'Compatibility',
		'is_active': 'Active',
		'tenant_id': 'Tenant',
		'created_at': 'Created',
		'updated_at': 'Updated',
		'created_by': 'Created By'
	}
	
	# Formatters
	formatters_columns = {
		'schema_definition': lambda x: json.dumps(x, indent=2) if x else "{}"
	}

# =============================================================================
# Metrics and Monitoring Views
# =============================================================================

class MetricsView(ModelView):
	"""View and analyze streaming metrics."""
	
	datamodel = SQLAInterface(ESMetrics)
	list_title = "Streaming Metrics"
	show_title = "Metric Details"
	
	# List view configuration
	list_columns = [
		'metric_name', 'metric_type', 'metric_value', 'metric_unit',
		'stream_id', 'consumer_group_id', 'time_bucket', 'aggregation_period'
	]
	
	search_columns = [
		'metric_name', 'metric_type', 'stream_id', 'consumer_group_id'
	]
	
	order_columns = ['time_bucket', 'metric_name', 'metric_value']
	base_order = ('time_bucket', 'desc')
	
	# Show view configuration
	show_columns = [
		'metric_id', 'metric_name', 'metric_type', 'stream_id',
		'consumer_group_id', 'metric_value', 'metric_unit', 'dimensions',
		'time_bucket', 'aggregation_period', 'tenant_id', 'created_at'
	]
	
	# No add/edit - metrics are system generated
	can_create = False
	can_edit = False
	can_delete = True  # Allow cleanup of old metrics
	
	# Labels
	label_columns = {
		'metric_id': 'Metric ID',
		'metric_name': 'Name',
		'metric_type': 'Type',
		'stream_id': 'Stream',
		'consumer_group_id': 'Consumer Group',
		'metric_value': 'Value',
		'metric_unit': 'Unit',
		'dimensions': 'Dimensions',
		'time_bucket': 'Time Bucket',
		'aggregation_period': 'Period',
		'tenant_id': 'Tenant',
		'created_at': 'Created'
	}
	
	# Formatters
	formatters_columns = {
		'dimensions': lambda x: json.dumps(x, indent=2) if x else "{}",
		'time_bucket': lambda x: x.strftime('%Y-%m-%d %H:%M:%S UTC') if x else "",
		'metric_value': lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else str(x)
	}

# =============================================================================
# Real-time Streaming Dashboard
# =============================================================================

class StreamingDashboardView(BaseView):
	"""Real-time dashboard for monitoring event streaming."""
	
	route_base = '/dashboard'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main streaming dashboard."""
		# Get summary statistics
		stats = {
			'total_streams': ESStream.query.count(),
			'active_streams': ESStream.query.filter_by(status=StreamStatus.ACTIVE.value).count(),
			'total_subscriptions': ESSubscription.query.count(),
			'active_subscriptions': ESSubscription.query.filter_by(status=SubscriptionStatus.ACTIVE.value).count(),
			'total_consumer_groups': ESConsumerGroup.query.count(),
			'events_today': ESEvent.query.filter(
				ESEvent.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
			).count()
		}
		
		# Get recent events
		recent_events = ESEvent.query.order_by(ESEvent.created_at.desc()).limit(10).all()
		
		# Get stream metrics
		stream_metrics = ESMetrics.query.filter(
			ESMetrics.metric_name == 'events_per_second'
		).filter(
			ESMetrics.time_bucket >= datetime.utcnow() - timedelta(hours=1)
		).order_by(ESMetrics.time_bucket.desc()).limit(20).all()
		
		return self.render_template(
			'streaming_dashboard.html',
			stats=stats,
			recent_events=recent_events,
			stream_metrics=stream_metrics,
			title="Event Streaming Dashboard"
		)
	
	@expose('/api/live_metrics')
	@has_access
	def live_metrics(self):
		"""API endpoint for live metrics data."""
		# Get latest metrics for dashboard
		metrics = {}
		
		# Events per second across all streams
		events_per_sec = ESMetrics.query.filter_by(metric_name='events_per_second')\
			.filter(ESMetrics.time_bucket >= datetime.utcnow() - timedelta(minutes=5))\
			.order_by(ESMetrics.time_bucket.desc()).first()
		
		metrics['events_per_second'] = events_per_sec.metric_value if events_per_sec else 0
		
		# Consumer lag
		consumer_lag = ESMetrics.query.filter_by(metric_name='consumer_lag')\
			.filter(ESMetrics.time_bucket >= datetime.utcnow() - timedelta(minutes=5))\
			.order_by(ESMetrics.time_bucket.desc()).first()
		
		metrics['consumer_lag'] = consumer_lag.metric_value if consumer_lag else 0
		
		# Stream status counts
		metrics['stream_status'] = {
			'active': ESStream.query.filter_by(status=StreamStatus.ACTIVE.value).count(),
			'paused': ESStream.query.filter_by(status=StreamStatus.PAUSED.value).count(),
			'error': ESStream.query.filter_by(status=StreamStatus.ERROR.value).count()
		}
		
		# Event status distribution
		metrics['event_status'] = {
			'published': ESEvent.query.filter_by(status=EventStatus.PUBLISHED.value).count(),
			'consumed': ESEvent.query.filter_by(status=EventStatus.CONSUMED.value).count(),
			'failed': ESEvent.query.filter_by(status=EventStatus.FAILED.value).count(),
			'retry': ESEvent.query.filter_by(status=EventStatus.RETRY.value).count()
		}
		
		return jsonify(metrics)
	
	@expose('/stream_topology')
	@has_access
	def stream_topology(self):
		"""Visualize stream topology and data flows."""
		# Get all streams and their relationships
		streams = ESStream.query.all()
		subscriptions = ESSubscription.query.all()
		
		# Build topology data structure
		topology = {
			'nodes': [],
			'edges': []
		}
		
		# Add stream nodes
		for stream in streams:
			topology['nodes'].append({
				'id': stream.stream_id,
				'label': stream.stream_name,
				'type': 'stream',
				'status': stream.status,
				'group': stream.source_capability
			})
		
		# Add subscription edges
		for sub in subscriptions:
			topology['edges'].append({
				'from': sub.stream_id,
				'to': sub.consumer_group_id,
				'label': sub.subscription_name,
				'status': sub.status
			})
		
		return self.render_template(
			'stream_topology.html',
			topology=topology,
			title="Stream Topology"
		)
	
	@expose('/event_browser')
	@has_access
	def event_browser(self):
		"""Real-time event browser with filtering."""
		# Get filter parameters
		stream_id = request.args.get('stream_id')
		event_type = request.args.get('event_type')
		start_time = request.args.get('start_time')
		end_time = request.args.get('end_time')
		
		# Build query
		query = ESEvent.query
		
		if stream_id:
			query = query.filter_by(stream_id=stream_id)
		if event_type:
			query = query.filter(ESEvent.event_type.like(f'%{event_type}%'))
		if start_time:
			query = query.filter(ESEvent.timestamp >= datetime.fromisoformat(start_time))
		if end_time:
			query = query.filter(ESEvent.timestamp <= datetime.fromisoformat(end_time))
		
		events = query.order_by(ESEvent.timestamp.desc()).limit(100).all()
		
		# Get available streams for filter dropdown
		streams = ESStream.query.all()
		
		return self.render_template(
			'event_browser.html',
			events=events,
			streams=streams,
			filters={
				'stream_id': stream_id,
				'event_type': event_type,
				'start_time': start_time,
				'end_time': end_time
			},
			title="Event Browser"
		)

# =============================================================================
# Enhanced Event Schema Registry Views
# =============================================================================

class EventSchemaView(ModelView):
	"""Enhanced event schema registry management."""
	
	datamodel = SQLAInterface(ESEventSchema)
	list_title = "Event Schema Registry"
	show_title = "Event Schema Details"
	add_title = "Register Event Schema"
	edit_title = "Edit Event Schema"
	
	# List view configuration
	list_columns = [
		'schema_name', 'schema_version', 'event_type', 'compatibility_level',
		'is_active', 'evolution_strategy', 'created_at'
	]
	
	search_columns = [
		'schema_name', 'event_type', 'compatibility_level', 'evolution_strategy'
	]
	
	order_columns = ['schema_name', 'schema_version', 'created_at']
	base_order = ('created_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'schema_id', 'schema_name', 'schema_version', 'json_schema',
		'event_type', 'compatibility_level', 'evolution_strategy',
		'validation_rules', 'is_active', 'tenant_id',
		'created_at', 'updated_at', 'created_by'
	]
	
	# Form configuration
	add_columns = [
		'schema_name', 'schema_version', 'json_schema', 'event_type',
		'compatibility_level', 'evolution_strategy', 'validation_rules', 'is_active'
	]
	
	edit_columns = [
		'json_schema', 'compatibility_level', 'evolution_strategy',
		'validation_rules', 'is_active'
	]
	
	# Field customization
	add_form_extra_fields = {
		'json_schema': TextAreaField(
			'JSON Schema',
			widget=JSONTextAreaWidget(),
			description='JSON Schema definition for event validation',
			validators=[DataRequired()]
		),
		'validation_rules': TextAreaField(
			'Validation Rules',
			widget=JSONTextAreaWidget(),
			description='Additional validation rules in JSON format'
		)
	}
	
	edit_form_extra_fields = add_form_extra_fields
	
	# Labels
	label_columns = {
		'schema_id': 'Schema ID',
		'schema_name': 'Schema Name',
		'schema_version': 'Version',
		'json_schema': 'JSON Schema',
		'event_type': 'Event Type',
		'compatibility_level': 'Compatibility',
		'evolution_strategy': 'Evolution Strategy',
		'validation_rules': 'Validation Rules',
		'is_active': 'Active',
		'tenant_id': 'Tenant',
		'created_at': 'Created',
		'updated_at': 'Updated',
		'created_by': 'Created By'
	}
	
	# Formatters
	formatters_columns = {
		'json_schema': lambda x: json.dumps(x, indent=2)[:200] + "..." if x else "{}",
		'validation_rules': lambda x: json.dumps(x, indent=2) if x else "{}"
	}

# =============================================================================
# Stream Assignment Management Views
# =============================================================================

class StreamAssignmentView(ModelView):
	"""Manage event-to-stream assignments."""
	
	datamodel = SQLAInterface(ESStreamAssignment)
	list_title = "Stream Assignments"
	show_title = "Assignment Details"
	add_title = "Create Stream Assignment"
	edit_title = "Edit Stream Assignment"
	
	# List view configuration
	list_columns = [
		'event_id', 'stream_id', 'assignment_type', 'priority_level',
		'is_active', 'assigned_at'
	]
	
	search_columns = [
		'event_id', 'stream_id', 'assignment_type', 'priority_level'
	]
	
	order_columns = ['assigned_at', 'priority_level']
	base_order = ('assigned_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'assignment_id', 'event_id', 'stream_id', 'assignment_type',
		'priority_level', 'routing_key', 'partition_strategy',
		'assignment_rules', 'is_active', 'tenant_id',
		'assigned_at', 'updated_at', 'assigned_by'
	]
	
	# Form configuration
	add_columns = [
		'event_id', 'stream_id', 'assignment_type', 'priority_level',
		'routing_key', 'partition_strategy', 'assignment_rules', 'is_active'
	]
	
	edit_columns = [
		'assignment_type', 'priority_level', 'routing_key',
		'partition_strategy', 'assignment_rules', 'is_active'
	]
	
	# Field customization
	add_form_extra_fields = {
		'assignment_rules': TextAreaField(
			'Assignment Rules',
			widget=JSONTextAreaWidget(),
			description='JSON rules for event assignment logic'
		)
	}
	
	edit_form_extra_fields = add_form_extra_fields
	
	# Labels
	label_columns = {
		'assignment_id': 'Assignment ID',
		'event_id': 'Event',
		'stream_id': 'Stream',
		'assignment_type': 'Type',
		'priority_level': 'Priority',
		'routing_key': 'Routing Key',
		'partition_strategy': 'Partition Strategy',
		'assignment_rules': 'Assignment Rules',
		'is_active': 'Active',
		'tenant_id': 'Tenant',
		'assigned_at': 'Assigned',
		'updated_at': 'Updated',
		'assigned_by': 'Assigned By'
	}

# =============================================================================
# Event Processing History Views
# =============================================================================

class EventProcessingHistoryView(ModelView):
	"""Monitor event processing history and audit trails."""
	
	datamodel = SQLAInterface(ESEventProcessingHistory)
	list_title = "Event Processing History"
	show_title = "Processing Details"
	
	# List view configuration
	list_columns = [
		'event_id', 'processor_id', 'processing_stage', 'status',
		'processing_duration_ms', 'started_at', 'completed_at'
	]
	
	search_columns = [
		'event_id', 'processor_id', 'processing_stage', 'status'
	]
	
	order_columns = ['started_at', 'processing_duration_ms']
	base_order = ('started_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'history_id', 'event_id', 'processor_id', 'processing_stage',
		'status', 'input_data', 'output_data', 'error_details',
		'processing_duration_ms', 'retry_count', 'tenant_id',
		'started_at', 'completed_at', 'processed_by'
	]
	
	# No add/edit - history is system generated
	can_create = False
	can_edit = False
	can_delete = True  # Allow cleanup of old history
	
	# Labels
	label_columns = {
		'history_id': 'History ID',
		'event_id': 'Event',
		'processor_id': 'Processor',
		'processing_stage': 'Stage',
		'status': 'Status',
		'input_data': 'Input Data',
		'output_data': 'Output Data',
		'error_details': 'Error Details',
		'processing_duration_ms': 'Duration (ms)',
		'retry_count': 'Retries',
		'tenant_id': 'Tenant',
		'started_at': 'Started',
		'completed_at': 'Completed',
		'processed_by': 'Processed By'
	}
	
	# Formatters
	formatters_columns = {
		'input_data': lambda x: json.dumps(x, indent=2)[:200] + "..." if x else "{}",
		'output_data': lambda x: json.dumps(x, indent=2)[:200] + "..." if x else "{}",
		'error_details': lambda x: str(x)[:200] + "..." if x and len(str(x)) > 200 else x or "",
		'processing_duration_ms': lambda x: f"{x:,.0f} ms" if x else "N/A",
		'started_at': lambda x: x.strftime('%Y-%m-%d %H:%M:%S UTC') if x else "",
		'completed_at': lambda x: x.strftime('%Y-%m-%d %H:%M:%S UTC') if x else "In Progress"
	}

# =============================================================================
# Stream Processor Management Views
# =============================================================================

class StreamProcessorView(ModelView):
	"""Manage stream processing jobs and configurations."""
	
	datamodel = SQLAInterface(ESStreamProcessor)
	list_title = "Stream Processors"
	show_title = "Processor Details"
	add_title = "Create Stream Processor"
	edit_title = "Edit Stream Processor"
	
	# List view configuration
	list_columns = [
		'processor_name', 'processor_type', 'source_stream_id',
		'target_stream_id', 'status', 'parallelism', 'created_at'
	]
	
	search_columns = [
		'processor_name', 'processor_type', 'source_stream_id', 'target_stream_id'
	]
	
	order_columns = ['processor_name', 'created_at', 'status']
	base_order = ('created_at', 'desc')
	
	# Show view configuration
	show_columns = [
		'processor_id', 'processor_name', 'processor_description',
		'processor_type', 'source_stream_id', 'target_stream_id',
		'processing_logic', 'state_store_config', 'window_config',
		'parallelism', 'error_handling_strategy', 'status',
		'tenant_id', 'created_at', 'updated_at', 'created_by'
	]
	
	# Form configuration
	add_columns = [
		'processor_name', 'processor_description', 'processor_type',
		'source_stream_id', 'target_stream_id', 'processing_logic',
		'state_store_config', 'window_config', 'parallelism',
		'error_handling_strategy'
	]
	
	edit_columns = [
		'processor_description', 'processing_logic', 'state_store_config',
		'window_config', 'parallelism', 'error_handling_strategy', 'status'
	]
	
	# Field customization
	add_form_extra_fields = {
		'processing_logic': TextAreaField(
			'Processing Logic',
			widget=JSONTextAreaWidget(),
			description='JSON configuration for processing logic',
			validators=[DataRequired()]
		),
		'state_store_config': TextAreaField(
			'State Store Configuration',
			widget=JSONTextAreaWidget(),
			description='Configuration for stateful processing'
		),
		'window_config': TextAreaField(
			'Window Configuration',
			widget=JSONTextAreaWidget(),
			description='Time window configuration for aggregations'
		)
	}
	
	edit_form_extra_fields = add_form_extra_fields
	
	# Labels
	label_columns = {
		'processor_id': 'Processor ID',
		'processor_name': 'Name',
		'processor_description': 'Description',
		'processor_type': 'Type',
		'source_stream_id': 'Source Stream',
		'target_stream_id': 'Target Stream',
		'processing_logic': 'Processing Logic',
		'state_store_config': 'State Store Config',
		'window_config': 'Window Config',
		'parallelism': 'Parallelism',
		'error_handling_strategy': 'Error Handling',
		'status': 'Status',
		'tenant_id': 'Tenant',
		'created_at': 'Created',
		'updated_at': 'Updated',
		'created_by': 'Created By'
	}
	
	# Formatters
	formatters_columns = {
		'processing_logic': lambda x: json.dumps(x, indent=2)[:200] + "..." if x else "{}",
		'state_store_config': lambda x: json.dumps(x, indent=2) if x else "{}",
		'window_config': lambda x: json.dumps(x, indent=2) if x else "{}"
	}
	
	@expose('/start_processor/<processor_id>')
	@has_access
	def start_processor(self, processor_id):
		"""Start stream processor."""
		try:
			processor = self.datamodel.get(processor_id)
			if processor:
				processor.status = 'RUNNING'
				self.datamodel.edit(processor)
				flash(f'Processor {processor.processor_name} started successfully', 'success')
			else:
				flash('Processor not found', 'error')
		except Exception as e:
			flash(f'Error starting processor: {str(e)}', 'error')
		
		return self.list()
	
	@expose('/stop_processor/<processor_id>')
	@has_access
	def stop_processor(self, processor_id):
		"""Stop stream processor."""
		try:
			processor = self.datamodel.get(processor_id)
			if processor:
				processor.status = 'STOPPED'
				self.datamodel.edit(processor)
				flash(f'Processor {processor.processor_name} stopped successfully', 'success')
			else:
				flash('Processor not found', 'error')
		except Exception as e:
			flash(f'Error stopping processor: {str(e)}', 'error')
		
		return self.list()

# =============================================================================
# Enhanced Streaming Dashboard with New Models
# =============================================================================

class EnhancedStreamingDashboardView(BaseView):
	"""Enhanced real-time dashboard with enterprise features."""
	
	route_base = '/enhanced_dashboard'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Enhanced streaming dashboard with new models."""
		# Enhanced statistics
		stats = {
			# Basic stats
			'total_streams': ESStream.query.count(),
			'active_streams': ESStream.query.filter_by(status=StreamStatus.ACTIVE.value).count(),
			'total_subscriptions': ESSubscription.query.count(),
			'active_subscriptions': ESSubscription.query.filter_by(status=SubscriptionStatus.ACTIVE.value).count(),
			'total_consumer_groups': ESConsumerGroup.query.count(),
			'events_today': ESEvent.query.filter(
				ESEvent.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
			).count(),
			
			# Enhanced stats
			'total_schemas': ESEventSchema.query.count(),
			'active_schemas': ESEventSchema.query.filter_by(is_active=True).count(),
			'total_processors': ESStreamProcessor.query.count(),
			'running_processors': ESStreamProcessor.query.filter_by(status='RUNNING').count(),
			'processing_history_count': ESEventProcessingHistory.query.count(),
			'stream_assignments': ESStreamAssignment.query.filter_by(is_active=True).count(),
			
			# Priority distribution
			'high_priority_events': ESEvent.query.filter_by(priority=EventPriority.HIGH.value).count(),
			'normal_priority_events': ESEvent.query.filter_by(priority=EventPriority.NORMAL.value).count(),
			'low_priority_events': ESEvent.query.filter_by(priority=EventPriority.LOW.value).count()
		}
		
		# Recent processing history
		recent_processing = ESEventProcessingHistory.query.order_by(
			ESEventProcessingHistory.started_at.desc()
		).limit(10).all()
		
		# Active processors
		active_processors = ESStreamProcessor.query.filter_by(status='RUNNING').all()
		
		# Schema registry status
		schema_stats = {
			'compatibility_levels': {},
			'evolution_strategies': {}
		}
		
		# Get schema distribution
		schemas = ESEventSchema.query.filter_by(is_active=True).all()
		for schema in schemas:
			level = schema.compatibility_level
			strategy = schema.evolution_strategy
			schema_stats['compatibility_levels'][level] = schema_stats['compatibility_levels'].get(level, 0) + 1
			schema_stats['evolution_strategies'][strategy] = schema_stats['evolution_strategies'].get(strategy, 0) + 1
		
		return self.render_template(
			'enhanced_streaming_dashboard.html',
			stats=stats,
			recent_processing=recent_processing,
			active_processors=active_processors,
			schema_stats=schema_stats,
			title="Enhanced Event Streaming Dashboard"
		)
	
	@expose('/processor_metrics')
	@has_access
	def processor_metrics(self):
		"""Display stream processor metrics."""
		processors = ESStreamProcessor.query.all()
		
		# Get processing metrics for each processor
		processor_metrics = []
		for processor in processors:
			# Get recent processing history
			history = ESEventProcessingHistory.query.filter_by(processor_id=processor.processor_id)\
				.filter(ESEventProcessingHistory.started_at >= datetime.utcnow() - timedelta(hours=24))\
				.all()
			
			# Calculate metrics
			total_processed = len(history)
			successful = len([h for h in history if h.status == 'COMPLETED'])
			failed = len([h for h in history if h.status == 'FAILED'])
			avg_duration = sum(h.processing_duration_ms or 0 for h in history) / total_processed if total_processed > 0 else 0
			
			processor_metrics.append({
				'processor': processor,
				'total_processed': total_processed,
				'successful': successful,
				'failed': failed,
				'success_rate': successful / total_processed if total_processed > 0 else 0,
				'avg_duration_ms': avg_duration
			})
		
		return self.render_template(
			'processor_metrics.html',
			processor_metrics=processor_metrics,
			title="Stream Processor Metrics"
		)
	
	@expose('/schema_registry')
	@has_access
	def schema_registry(self):
		"""Schema registry dashboard."""
		schemas = ESEventSchema.query.all()
		
		# Group schemas by event type
		schema_groups = {}
		for schema in schemas:
			event_type = schema.event_type
			if event_type not in schema_groups:
				schema_groups[event_type] = []
			schema_groups[event_type].append(schema)
		
		# Sort schemas within each group by version
		for event_type in schema_groups:
			schema_groups[event_type].sort(key=lambda s: s.schema_version, reverse=True)
		
		return self.render_template(
			'schema_registry.html',
			schema_groups=schema_groups,
			title="Schema Registry Dashboard"
		)

# Export all views
__all__ = [
	'EventStreamView',
	'EventView', 
	'SubscriptionView',
	'ConsumerGroupView',
	'SchemaView',
	'MetricsView',
	'StreamingDashboardView',
	'EventSchemaView',
	'StreamAssignmentView',
	'EventProcessingHistoryView',
	'StreamProcessorView',
	'EnhancedStreamingDashboardView'
]
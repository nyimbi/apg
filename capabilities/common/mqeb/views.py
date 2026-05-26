#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Flask-AppBuilder Views
Flask-AppBuilder views for MQEB management interface

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from flask import Blueprint, request, jsonify, g
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from .models import (
	MQMessage, TopicConfiguration, Subscription, MessageEvent, 
	BrokerNode, MessagePriority, DeliveryMode, ProtocolType
)

# Create Flask blueprint for MQEB
mqeb_bp = Blueprint('mqeb', __name__, url_prefix='/mqeb')


class MQEBDashboardView(BaseView):
	"""Main dashboard view for MQEB"""
	
	default_view = 'dashboard'
	
	@expose('/dashboard/')
	@has_access
	def dashboard(self):
		"""Main MQEB dashboard"""
		
		# Get cluster statistics
		cluster_stats = self._get_cluster_stats()
		
		# Get recent message activity
		recent_activity = self._get_recent_activity()
		
		# Get performance metrics
		performance_metrics = self._get_performance_metrics()
		
		# Get alerts and health status
		health_status = self._get_health_status()
		
		return self.render_template(
			'mqeb/dashboard.html',
			cluster_stats=cluster_stats,
			recent_activity=recent_activity,
			performance_metrics=performance_metrics,
			health_status=health_status
		)
	
	@expose('/api/stats')
	@has_access
	def api_stats(self):
		"""API endpoint for dashboard statistics"""
		
		stats = {
			'cluster': self._get_cluster_stats(),
			'performance': self._get_performance_metrics(),
			'health': self._get_health_status()
		}
		
		return jsonify(stats)
	
	def _get_cluster_stats(self) -> Dict[str, Any]:
		"""Get cluster-wide statistics"""
		
		# In production, would query actual database/metrics
		return {
			'total_topics': 125,
			'total_partitions': 850,
			'total_subscriptions': 350,
			'active_brokers': 5,
			'total_messages_today': 15750000,
			'messages_per_second': 45000,
			'average_latency_ms': 2.3,
			'active_connections': 2850,
			'data_processed_gb': 125.7
		}
	
	def _get_recent_activity(self) -> List[Dict[str, Any]]:
		"""Get recent message activity"""
		
		# In production, would query actual message events
		return [
			{
				'timestamp': '2025-01-09T10:25:00Z',
				'event': 'Topic Created',
				'topic': 'user.events.login',
				'user': 'admin@datacraft.co.ke',
				'details': 'Created with 10 partitions'
			},
			{
				'timestamp': '2025-01-09T10:20:00Z',
				'event': 'High Message Volume',
				'topic': 'system.metrics',
				'details': '50K messages/sec spike detected'
			},
			{
				'timestamp': '2025-01-09T10:15:00Z',
				'event': 'Subscription Created',
				'topic': 'notifications.*',
				'user': 'service@example.com',
				'details': 'WebSocket subscription'
			}
		]
	
	def _get_performance_metrics(self) -> Dict[str, Any]:
		"""Get performance metrics"""
		
		return {
			'throughput': {
				'messages_per_second': 45000,
				'bytes_per_second': 125000000,
				'peak_messages_per_second': 78000
			},
			'latency': {
				'p50_ms': 1.2,
				'p90_ms': 3.1,
				'p99_ms': 4.8,
				'max_ms': 12.5
			},
			'resources': {
				'cpu_usage': 45.2,
				'memory_usage': 62.8,
				'disk_usage': 35.1,
				'network_io_mbps': 850.3
			}
		}
	
	def _get_health_status(self) -> Dict[str, Any]:
		"""Get system health status"""
		
		return {
			'overall_status': 'healthy',
			'broker_cluster': 'healthy',
			'message_routing': 'healthy', 
			'storage_system': 'healthy',
			'protocol_adapters': {
				'http_rest': 'healthy',
				'websocket': 'healthy',
				'mqtt': 'healthy',
				'amqp': 'degraded',  # Example degraded service
				'bytewax': 'healthy'
			},
			'alerts': [
				{
					'severity': 'warning',
					'component': 'amqp_adapter',
					'message': 'Connection pool utilization at 85%',
					'timestamp': '2025-01-09T10:20:00Z'
				}
			]
		}


class TopicManagementView(BaseView):
	"""Topic management view"""
	
	@expose('/topics/')
	@has_access
	def list_topics(self):
		"""List all topics"""
		
		# Get topics for current tenant
		tenant_id = getattr(g, 'tenant_id', 'default')
		topics = self._get_tenant_topics(tenant_id)
		
		return self.render_template(
			'mqeb/topics/list.html',
			topics=topics
		)
	
	@expose('/topics/create/', methods=['GET', 'POST'])
	@has_access
	def create_topic(self):
		"""Create new topic"""
		
		if request.method == 'POST':
			# Process topic creation
			topic_data = self._process_topic_form()
			if topic_data:
				# Would create actual topic in production
				return self.redirect('/mqeb/topics/')
		
		# Show topic creation form
		return self.render_template('mqeb/topics/create.html')
	
	@expose('/topics/<topic_name>/')
	@has_access
	def show_topic(self, topic_name):
		"""Show topic details"""
		
		topic_info = self._get_topic_details(topic_name)
		topic_metrics = self._get_topic_metrics(topic_name)
		recent_messages = self._get_recent_messages(topic_name)
		
		return self.render_template(
			'mqeb/topics/show.html',
			topic=topic_info,
			metrics=topic_metrics,
			recent_messages=recent_messages
		)
	
	@expose('/api/topics')
	@has_access
	def api_list_topics(self):
		"""API endpoint to list topics"""
		
		tenant_id = getattr(g, 'tenant_id', 'default')
		topics = self._get_tenant_topics(tenant_id)
		
		return jsonify({
			'topics': topics,
			'total': len(topics)
		})
	
	def _get_tenant_topics(self, tenant_id: str) -> List[Dict[str, Any]]:
		"""Get topics for specific tenant"""
		
		# In production, would query database
		return [
			{
				'name': 'user.events.login',
				'partitions': 10,
				'messages_today': 125000,
				'size_mb': 145.2,
				'retention_days': 7,
				'encryption': True,
				'created_at': '2025-01-05T14:30:00Z'
			},
			{
				'name': 'system.metrics',
				'partitions': 20,
				'messages_today': 850000,
				'size_mb': 1250.7,
				'retention_days': 30,
				'encryption': True,
				'created_at': '2025-01-01T09:00:00Z'
			}
		]
	
	def _get_topic_details(self, topic_name: str) -> Dict[str, Any]:
		"""Get detailed topic information"""
		
		return {
			'name': topic_name,
			'display_name': topic_name.replace('.', ' ').title(),
			'description': f'Topic for {topic_name} messages',
			'partitions': 10,
			'replication_factor': 3,
			'retention_ms': 604800000,  # 7 days
			'max_message_size': 1048576,  # 1MB
			'compression_type': 'snappy',
			'encryption_required': True,
			'created_at': '2025-01-05T14:30:00Z',
			'updated_at': '2025-01-08T16:45:00Z'
		}
	
	def _get_topic_metrics(self, topic_name: str) -> Dict[str, Any]:
		"""Get topic performance metrics"""
		
		return {
			'messages_per_second': 2500,
			'bytes_per_second': 3500000,
			'producer_count': 5,
			'consumer_count': 12,
			'partition_distribution': [
				{'partition': 0, 'messages': 12500, 'size_mb': 15.2},
				{'partition': 1, 'messages': 11800, 'size_mb': 14.5},
				# ... more partitions
			]
		}
	
	def _get_recent_messages(self, topic_name: str, limit: int = 10) -> List[Dict[str, Any]]:
		"""Get recent messages from topic"""
		
		return [
			{
				'id': 'msg_12345',
				'timestamp': '2025-01-09T10:25:30Z',
				'size_bytes': 1024,
				'producer': 'user-service',
				'partition': 5,
				'headers': {'event-type': 'login', 'user-id': '12345'}
			}
		]
	
	def _process_topic_form(self) -> Optional[Dict[str, Any]]:
		"""Process topic creation form"""
		
		try:
			form_data = request.get_json() if request.is_json else request.form
			
			# Validate and process form data
			topic_config = {
				'name': form_data.get('name'),
				'partitions': int(form_data.get('partitions', 1)),
				'replication_factor': int(form_data.get('replication_factor', 3)),
				'retention_ms': int(form_data.get('retention_days', 7)) * 86400000,
				'encryption_required': form_data.get('encryption_required', True),
				'tenant_id': getattr(g, 'tenant_id', 'default'),
				'created_by': getattr(g, 'user_id', 'anonymous')
			}
			
			# Would validate and create topic in production
			return topic_config
			
		except Exception as e:
			print(f"Error processing topic form: {e}")
			return None


class MessagePublishingView(BaseView):
	"""Message publishing interface"""
	
	@expose('/publish/')
	@has_access
	def publish_form(self):
		"""Show message publishing form"""
		
		tenant_id = getattr(g, 'tenant_id', 'default')
		available_topics = self._get_tenant_topics(tenant_id)
		
		return self.render_template(
			'mqeb/messages/publish.html',
			topics=available_topics
		)
	
	@expose('/api/publish', methods=['POST'])
	@has_access
	def api_publish(self):
		"""API endpoint for publishing messages"""
		
		try:
			message_data = request.get_json()
			
			# Validate message data
			if not message_data or not message_data.get('topic'):
				return jsonify({'error': 'Topic is required'}), 400
			
			# Process message publishing
			result = self._publish_message(message_data)
			
			if result['success']:
				return jsonify(result), 200
			else:
				return jsonify(result), 400
				
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	def _publish_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Publish message to topic"""
		
		try:
			# In production, would use actual MQEB service
			message_id = f"msg_{int(datetime.utcnow().timestamp())}"
			
			# Simulate message publishing
			result = {
				'success': True,
				'message_id': message_id,
				'topic': message_data['topic'],
				'partition': 0,  # Would be calculated
				'offset': 12345,  # Would be actual offset
				'timestamp': datetime.utcnow().isoformat(),
				'size_bytes': len(str(message_data.get('payload', '')))
			}
			
			print(f"[MQEB] Published message {message_id} to topic {message_data['topic']}")
			
			return result
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}
	
	def _get_tenant_topics(self, tenant_id: str) -> List[str]:
		"""Get available topics for tenant"""
		
		# In production, would query database
		return [
			'user.events.login',
			'user.events.logout', 
			'system.metrics',
			'notifications.email',
			'workflows.triggers'
		]


class SubscriptionManagementView(BaseView):
	"""Subscription management interface"""
	
	@expose('/subscriptions/')
	@has_access
	def list_subscriptions(self):
		"""List all subscriptions"""
		
		tenant_id = getattr(g, 'tenant_id', 'default')
		subscriptions = self._get_tenant_subscriptions(tenant_id)
		
		return self.render_template(
			'mqeb/subscriptions/list.html',
			subscriptions=subscriptions
		)
	
	@expose('/subscriptions/create/', methods=['GET', 'POST'])
	@has_access
	def create_subscription(self):
		"""Create new subscription"""
		
		if request.method == 'POST':
			# Process subscription creation
			subscription_data = self._process_subscription_form()
			if subscription_data:
				# Would create actual subscription in production
				return self.redirect('/mqeb/subscriptions/')
		
		# Show subscription creation form
		available_topics = self._get_tenant_topics(getattr(g, 'tenant_id', 'default'))
		
		return self.render_template(
			'mqeb/subscriptions/create.html',
			topics=available_topics,
			protocols=list(ProtocolType),
			delivery_modes=list(DeliveryMode)
		)
	
	def _get_tenant_subscriptions(self, tenant_id: str) -> List[Dict[str, Any]]:
		"""Get subscriptions for tenant"""
		
		return [
			{
				'id': 'sub_12345',
				'name': 'Email Notifications',
				'topic_pattern': 'notifications.email.*',
				'protocol': 'http_rest',
				'webhook_url': 'https://api.example.com/webhooks/email',
				'status': 'active',
				'messages_delivered': 15000,
				'success_rate': 99.8,
				'created_at': '2025-01-05T14:30:00Z'
			}
		]
	
	def _process_subscription_form(self) -> Optional[Dict[str, Any]]:
		"""Process subscription creation form"""
		
		try:
			form_data = request.get_json() if request.is_json else request.form
			
			subscription_config = {
				'name': form_data.get('name'),
				'topic_pattern': form_data.get('topic_pattern'),
				'protocol': form_data.get('protocol', 'http_rest'),
				'webhook_url': form_data.get('webhook_url'),
				'delivery_mode': form_data.get('delivery_mode', 'at_least_once'),
				'tenant_id': getattr(g, 'tenant_id', 'default'),
				'created_by': getattr(g, 'user_id', 'anonymous')
			}
			
			return subscription_config
			
		except Exception as e:
			print(f"Error processing subscription form: {e}")
			return None


class MonitoringView(BaseView):
	"""Monitoring and metrics view"""
	
	@expose('/monitor/performance/')
	@has_access
	def performance_monitor(self):
		"""Performance monitoring dashboard"""
		
		metrics = self._get_performance_metrics()
		
		return self.render_template(
			'mqeb/monitoring/performance.html',
			metrics=metrics
		)
	
	@expose('/monitor/cluster/')
	@has_access
	def cluster_monitor(self):
		"""Cluster health monitoring"""
		
		cluster_health = self._get_cluster_health()
		
		return self.render_template(
			'mqeb/monitoring/cluster.html',
			cluster=cluster_health
		)
	
	@expose('/api/metrics')
	@has_access
	def api_metrics(self):
		"""API endpoint for metrics"""
		
		metrics = {
			'timestamp': datetime.utcnow().isoformat(),
			'throughput': {
				'messages_per_second': 45000,
				'bytes_per_second': 125000000
			},
			'latency': {
				'p50_ms': 1.2,
				'p99_ms': 4.8
			},
			'resources': {
				'cpu_usage': 45.2,
				'memory_usage': 62.8
			}
		}
		
		return jsonify(metrics)
	
	def _get_performance_metrics(self) -> Dict[str, Any]:
		"""Get detailed performance metrics"""
		
		return {
			'current_timestamp': datetime.utcnow().isoformat(),
			'throughput_metrics': {
				'messages_per_second': 45000,
				'bytes_per_second': 125000000,
				'peak_messages_per_second': 78000,
				'total_messages_today': 15750000
			},
			'latency_metrics': {
				'p50_ms': 1.2,
				'p90_ms': 3.1,
				'p99_ms': 4.8,
				'max_ms': 12.5,
				'average_ms': 2.3
			},
			'error_metrics': {
				'error_rate': 0.02,
				'timeout_rate': 0.01,
				'dead_letter_messages': 125
			}
		}
	
	def _get_cluster_health(self) -> Dict[str, Any]:
		"""Get cluster health information"""
		
		return {
			'cluster_status': 'healthy',
			'total_nodes': 5,
			'healthy_nodes': 5,
			'leader_node': 'mqeb-broker-01',
			'nodes': [
				{
					'id': 'mqeb-broker-01',
					'hostname': 'broker-01.mqeb.local',
					'status': 'healthy',
					'cpu_usage': 45.2,
					'memory_usage': 62.8,
					'partitions_hosted': 170,
					'connections': 580
				}
			]
		}


def init_views(appbuilder):
	"""Initialize all MQEB views"""
	
	# Add main views
	appbuilder.add_view(
		MQEBDashboardView,
		"Dashboard",
		icon="fa-tachometer-alt",
		category="MQEB",
		category_icon="fa-exchange-alt"
	)
	
	appbuilder.add_view(
		TopicManagementView,
		"Topics",
		icon="fa-list",
		category="MQEB"
	)
	
	appbuilder.add_view(
		MessagePublishingView,
		"Publish Messages",
		icon="fa-paper-plane",
		category="MQEB"
	)
	
	appbuilder.add_view(
		SubscriptionManagementView,
		"Subscriptions",
		icon="fa-rss",
		category="MQEB"
	)
	
	appbuilder.add_view(
		MonitoringView,
		"Monitoring",
		icon="fa-chart-line",
		category="MQEB"
	)


# Export components
__all__ = [
	'mqeb_bp',
	'init_views',
	'MQEBDashboardView',
	'TopicManagementView', 
	'MessagePublishingView',
	'SubscriptionManagementView',
	'MonitoringView'
]

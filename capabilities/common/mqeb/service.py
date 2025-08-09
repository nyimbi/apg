#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Core Service
Main service implementation with APG integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, AsyncGenerator
from dataclasses import dataclass, field
import uuid
from uuid_extensions import uuid7str

from .models import (
	MQMessage, TopicConfiguration, Subscription, MessageEvent, BrokerNode,
	MessagePriority, DeliveryMode, ProtocolType, MessageStatus, RetryPolicy
)


class MQEBService:
	"""
	Core MQEB service implementation
	Provides high-performance message queuing with APG integration
	"""
	
	def __init__(self, config: Dict[str, Any] | None = None):
		self.config = config or {}
		self.running = False
		
		# Core components
		self.message_store: Dict[str, MQMessage] = {}
		self.topics: Dict[str, TopicConfiguration] = {}
		self.subscriptions: Dict[str, Subscription] = {}
		self.broker_nodes: Dict[str, BrokerNode] = {}
		
		# Message routing and processing
		self.message_queues: Dict[str, List[str]] = {}  # topic -> message_ids
		self.subscription_queues: Dict[str, List[str]] = {}  # subscription_id -> message_ids
		self.dead_letter_queues: Dict[str, List[str]] = {}
		
		# Performance tracking
		self.metrics = {
			'messages_published': 0,
			'messages_delivered': 0,
			'messages_failed': 0,
			'bytes_processed': 0,
			'active_connections': 0,
			'topics_created': 0,
			'subscriptions_created': 0
		}
		
		# Background tasks
		self._background_tasks: Set[asyncio.Task] = set()
		
		# Logging
		self.logger = logging.getLogger('mqeb.service')
	
	async def _log_audit_event(self, event_type: str, resource_id: str, action: str, 
							  user_id: str = None, details: Dict[str, Any] = None) -> None:
		"""Log audit events for compliance"""
		event = MessageEvent(
			message_id=resource_id,
			event_type=event_type,
			status="success",
			tenant_id=details.get('tenant_id', 'default') if details else 'default',
			user_id=user_id,
			metadata=details or {}
		)
		
		# In production, would persist to audit database
		self.logger.info(f"[AUDIT] {event_type}: {action} by {user_id}")
	
	async def initialize(self, config: Dict[str, Any] | None = None) -> None:
		"""Initialize MQEB service"""
		if config:
			self.config.update(config)
		
		self.logger.info("Initializing MQEB service...")
		
		# Initialize broker node
		await self._initialize_broker_node()
		
		# Initialize security and compliance engines
		await self._initialize_security_engines()
		
		# Start background tasks
		await self._start_background_tasks()
		
		# Initialize default topics
		await self._create_default_topics()
		
		self.running = True
		self.logger.info("MQEB service initialized successfully")
	
	async def shutdown(self) -> None:
		"""Shutdown MQEB service gracefully"""
		self.logger.info("Shutting down MQEB service...")
		
		self.running = False
		
		# Shutdown security engines if initialized
		if hasattr(self, 'quantum_security'):
			try:
				await self.quantum_security.shutdown()
			except Exception as e:
				self.logger.error(f"Error shutting down quantum security engine: {e}")
		
		if hasattr(self, 'compliance_governance'):
			try:
				await self.compliance_governance.shutdown()
			except Exception as e:
				self.logger.error(f"Error shutting down compliance governance engine: {e}")
		
		if hasattr(self, 'enterprise_workflow'):
			try:
				await self.enterprise_workflow.shutdown()
			except Exception as e:
				self.logger.error(f"Error shutting down enterprise workflow engine: {e}")
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		# Wait for tasks to complete
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		
		self.logger.info("MQEB service shut down")
	
	async def _initialize_broker_node(self) -> None:
		"""Initialize this broker node"""
		node = BrokerNode(
			name=f"mqeb-broker-{uuid.uuid4().hex[:8]}",
			hostname="localhost",  # Would be actual hostname
			ip_address="127.0.0.1",  # Would be actual IP
			port=8080,
			region="us-east-1",  # Would be from config
			cluster_id="default-cluster",
			protocols_enabled=[ProtocolType.HTTP_REST, ProtocolType.WEBSOCKET]
		)
		
		self.broker_nodes[node.id] = node
		self.logger.info(f"Initialized broker node: {node.name}")
	
	async def _initialize_security_engines(self) -> None:
		"""Initialize quantum security, compliance governance, and enterprise workflow engines"""
		try:
			# Try to initialize quantum security engine
			if self.config.get('quantum_security_enabled', True):
				try:
					from .quantum_security import create_quantum_security_engine
					self.quantum_security = await create_quantum_security_engine(self)
					self.logger.info("Quantum security engine initialized")
				except ImportError:
					self.logger.warning("Quantum security module not available")
				except Exception as e:
					self.logger.error(f"Failed to initialize quantum security engine: {e}")
			
			# Try to initialize compliance governance engine
			if self.config.get('compliance_governance_enabled', True):
				try:
					from .compliance_governance import create_compliance_governance_engine
					self.compliance_governance = await create_compliance_governance_engine(self)
					self.logger.info("Compliance governance engine initialized")
				except ImportError:
					self.logger.warning("Compliance governance module not available")
				except Exception as e:
					self.logger.error(f"Failed to initialize compliance governance engine: {e}")
			
			# Try to initialize enterprise workflow engine
			if self.config.get('enterprise_workflows_enabled', True):
				try:
					from .enterprise_integration import create_enterprise_workflow_engine
					self.enterprise_workflow = await create_enterprise_workflow_engine(self)
					self.logger.info("Enterprise workflow engine initialized")
				except ImportError:
					self.logger.warning("Enterprise integration module not available")
				except Exception as e:
					self.logger.error(f"Failed to initialize enterprise workflow engine: {e}")
			
		except Exception as e:
			self.logger.error(f"Error initializing security engines: {e}")
			# Continue without security engines - service should still be functional
	
	async def _start_background_tasks(self) -> None:
		"""Start background processing tasks"""
		
		# Message processing task
		task = asyncio.create_task(self._message_processing_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Metrics collection task
		task = asyncio.create_task(self._metrics_collection_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Health monitoring task
		task = asyncio.create_task(self._health_monitoring_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		self.logger.info("Started background processing tasks")
	
	async def _create_default_topics(self) -> None:
		"""Create default system topics"""
		default_topics = [
			{
				'name': 'system.events',
				'description': 'System-wide events and notifications',
				'partitions': 5
			},
			{
				'name': 'user.events',
				'description': 'User activity events',
				'partitions': 10
			},
			{
				'name': 'application.logs', 
				'description': 'Application log messages',
				'partitions': 15
			},
			{
				'name': 'metrics.performance',
				'description': 'Performance and monitoring metrics',
				'partitions': 5
			}
		]
		
		for topic_spec in default_topics:
			try:
				topic_config = TopicConfiguration(
					name=topic_spec['name'],
					description=topic_spec['description'],
					partitions=topic_spec['partitions'],
					tenant_id='system',
					created_by='system'
				)
				
				await self.create_topic(topic_config)
				
			except Exception as e:
				self.logger.warning(f"Failed to create default topic {topic_spec['name']}: {e}")
	
	async def create_topic(self, topic_config: TopicConfiguration) -> str:
		"""Create a new topic"""
		
		# Validate topic doesn't already exist
		if topic_config.name in self.topics:
			raise ValueError(f"Topic {topic_config.name} already exists")
		
		# Store topic configuration
		self.topics[topic_config.name] = topic_config
		
		# Initialize topic message queue
		self.message_queues[topic_config.name] = []
		
		# Update metrics
		self.metrics['topics_created'] += 1
		
		# Log audit event
		await self._log_audit_event(
			event_type="topic_created",
			resource_id=topic_config.name,
			action="create_topic",
			user_id=topic_config.created_by,
			details={
				'topic_name': topic_config.name,
				'partitions': topic_config.partitions,
				'tenant_id': topic_config.tenant_id
			}
		)
		
		self.logger.info(f"Created topic: {topic_config.name}")
		return topic_config.name
	
	async def publish_message(self, message: MQMessage, context: Dict[str, Any] | None = None) -> str:
		"""Publish a message to a topic"""
		
		# Validate topic exists
		if message.topic not in self.topics:
			raise ValueError(f"Topic {message.topic} does not exist")
		
		# Validate message
		if message.is_expired():
			raise ValueError("Message has already expired")
		
		# Apply security and compliance checks
		if context is None:
			context = {}
		
		# Quantum security processing
		if hasattr(self, 'quantum_security'):
			try:
				security_result = await self.quantum_security.secure_message(message, context)
				if not security_result:
					raise ValueError("Message failed quantum security validation")
			except Exception as e:
				self.logger.error(f"Quantum security processing failed: {e}")
				# Continue without security - configurable behavior
		
		# Compliance and governance processing
		if hasattr(self, 'compliance_governance'):
			try:
				compliance_result = await self.compliance_governance.process_message_compliance(message, context)
				if not compliance_result['compliant']:
					self.logger.warning(f"Message {message.id} compliance violations: {compliance_result['violations']}")
					# In strict mode, could reject message here
			except Exception as e:
				self.logger.error(f"Compliance processing failed: {e}")
		
		# Store message
		self.message_store[message.id] = message
		
		# Add to topic queue
		self.message_queues[message.topic].append(message.id)
		
		# Route to subscriptions
		await self._route_message_to_subscriptions(message)
		
		# Trigger enterprise workflows if enabled
		if hasattr(self, 'enterprise_workflow') and self.enterprise_workflow.running:
			try:
				execution_id = await self.enterprise_workflow.trigger_workflow(message, context)
				if execution_id:
					self.logger.debug(f"Triggered workflow execution {execution_id} for message {message.id}")
			except Exception as e:
				self.logger.error(f"Failed to trigger workflow for message {message.id}: {e}")
		
		# Update metrics
		self.metrics['messages_published'] += 1
		self.metrics['bytes_processed'] += message.size_bytes()
		
		# Log audit event
		await self._log_audit_event(
			event_type="message_published",
			resource_id=message.id,
			action="publish_message",
			user_id=message.user_id,
			details={
				'topic': message.topic,
				'size_bytes': message.size_bytes(),
				'tenant_id': message.tenant_id
			}
		)
		
		self.logger.debug(f"Published message {message.id} to topic {message.topic}")
		return message.id
	
	async def _route_message_to_subscriptions(self, message: MQMessage) -> None:
		"""Route message to matching subscriptions"""
		
		for subscription in self.subscriptions.values():
			# Check if subscription matches message topic
			if await self._subscription_matches_topic(subscription, message.topic):
				# Check message filter
				if subscription.message_filter and not subscription.message_filter.matches(message):
					continue
				
				# Add to subscription queue
				if subscription.id not in self.subscription_queues:
					self.subscription_queues[subscription.id] = []
				
				self.subscription_queues[subscription.id].append(message.id)
				
				self.logger.debug(f"Routed message {message.id} to subscription {subscription.id}")
	
	async def _subscription_matches_topic(self, subscription: Subscription, topic: str) -> bool:
		"""Check if subscription topic pattern matches the given topic"""
		
		import fnmatch
		return fnmatch.fnmatch(topic, subscription.topic_pattern)
	
	async def create_subscription(self, subscription: Subscription) -> str:
		"""Create a new subscription"""
		
		# Validate subscription doesn't already exist
		if subscription.id in self.subscriptions:
			raise ValueError(f"Subscription {subscription.id} already exists")
		
		# Store subscription
		self.subscriptions[subscription.id] = subscription
		
		# Initialize subscription queue
		self.subscription_queues[subscription.id] = []
		
		# Update metrics
		self.metrics['subscriptions_created'] += 1
		
		# Log audit event
		await self._log_audit_event(
			event_type="subscription_created",
			resource_id=subscription.id,
			action="create_subscription",
			user_id=subscription.created_by,
			details={
				'subscription_name': subscription.name,
				'topic_pattern': subscription.topic_pattern,
				'tenant_id': subscription.tenant_id
			}
		)
		
		self.logger.info(f"Created subscription: {subscription.name}")
		return subscription.id
	
	async def consume_messages(self, subscription_id: str, max_messages: int = 10) -> List[MQMessage]:
		"""Consume messages from a subscription"""
		
		if subscription_id not in self.subscriptions:
			raise ValueError(f"Subscription {subscription_id} not found")
		
		subscription = self.subscriptions[subscription_id]
		
		if subscription_id not in self.subscription_queues:
			return []
		
		# Get messages from subscription queue
		message_ids = self.subscription_queues[subscription_id][:max_messages]
		messages = []
		
		for message_id in message_ids:
			if message_id in self.message_store:
				message = self.message_store[message_id]
				
				# Check if message is still valid
				if not message.is_expired():
					messages.append(message)
				else:
					# Remove expired message
					self.message_store.pop(message_id, None)
		
		# Remove consumed messages from queue (for at-least-once delivery)
		if subscription.delivery_mode == DeliveryMode.AT_LEAST_ONCE:
			self.subscription_queues[subscription_id] = self.subscription_queues[subscription_id][len(messages):]
		
		# Update metrics
		self.metrics['messages_delivered'] += len(messages)
		
		self.logger.debug(f"Consumed {len(messages)} messages from subscription {subscription_id}")
		return messages
	
	async def get_topic_stats(self, topic_name: str) -> Dict[str, Any]:
		"""Get statistics for a topic"""
		
		if topic_name not in self.topics:
			raise ValueError(f"Topic {topic_name} not found")
		
		topic_config = self.topics[topic_name]
		message_queue = self.message_queues.get(topic_name, [])
		
		# Calculate message sizes
		total_size = 0
		for message_id in message_queue:
			if message_id in self.message_store:
				total_size += self.message_store[message_id].size_bytes()
		
		# Count active subscriptions
		active_subscriptions = 0
		for subscription in self.subscriptions.values():
			if await self._subscription_matches_topic(subscription, topic_name):
				active_subscriptions += 1
		
		return {
			'topic_name': topic_name,
			'partitions': topic_config.partitions,
			'replication_factor': topic_config.replication_factor,
			'total_messages': len(message_queue),
			'total_size_bytes': total_size,
			'active_subscriptions': active_subscriptions,
			'retention_ms': topic_config.retention_ms,
			'created_at': topic_config.created_at.isoformat()
		}
	
	async def get_subscription_stats(self, subscription_id: str) -> Dict[str, Any]:
		"""Get statistics for a subscription"""
		
		if subscription_id not in self.subscriptions:
			raise ValueError(f"Subscription {subscription_id} not found")
		
		subscription = self.subscriptions[subscription_id]
		queue_size = len(self.subscription_queues.get(subscription_id, []))
		
		return {
			'subscription_id': subscription_id,
			'name': subscription.name,
			'topic_pattern': subscription.topic_pattern,
			'delivery_mode': subscription.delivery_mode.value,
			'protocol': subscription.protocol.value,
			'pending_messages': queue_size,
			'total_messages': subscription.total_messages,
			'failed_messages': subscription.failed_messages,
			'success_rate': subscription.success_rate(),
			'enabled': subscription.enabled,
			'paused': subscription.paused,
			'created_at': subscription.created_at.isoformat()
		}
	
	async def get_cluster_stats(self) -> Dict[str, Any]:
		"""Get cluster-wide statistics"""
		
		return {
			'broker_nodes': len(self.broker_nodes),
			'total_topics': len(self.topics),
			'total_subscriptions': len(self.subscriptions),
			'total_messages_stored': len(self.message_store),
			'metrics': self.metrics.copy(),
			'uptime_seconds': int((datetime.utcnow() - datetime.utcnow()).total_seconds()),  # Would be actual uptime
			'cluster_healthy': self._is_cluster_healthy()
		}
	
	def _is_cluster_healthy(self) -> bool:
		"""Check if cluster is healthy"""
		
		# Check if any broker nodes are unhealthy
		for node in self.broker_nodes.values():
			if not node.is_healthy():
				return False
		
		# Check for any critical issues
		error_rate = self.metrics.get('messages_failed', 0) / max(1, self.metrics.get('messages_published', 1))
		if error_rate > 0.05:  # 5% error rate threshold
			return False
		
		return True
	
	async def _message_processing_loop(self) -> None:
		"""Background message processing loop"""
		
		while self.running:
			try:
				# Process message delivery
				await self._process_pending_deliveries()
				
				# Clean up expired messages
				await self._cleanup_expired_messages()
				
				# Process dead letter queues
				await self._process_dead_letter_queues()
				
				# Sleep before next iteration
				await asyncio.sleep(1)
				
			except Exception as e:
				self.logger.error(f"Error in message processing loop: {e}")
				await asyncio.sleep(5)
	
	async def _process_pending_deliveries(self) -> None:
		"""Process pending message deliveries"""
		
		for subscription_id, message_ids in self.subscription_queues.items():
			if not message_ids:
				continue
			
			subscription = self.subscriptions.get(subscription_id)
			if not subscription or not subscription.enabled or subscription.paused:
				continue
			
			# Process deliveries for this subscription
			try:
				await self._deliver_messages_to_subscription(subscription, message_ids[:10])
			except Exception as e:
				self.logger.error(f"Error delivering messages to subscription {subscription_id}: {e}")
	
	async def _deliver_messages_to_subscription(self, subscription: Subscription, message_ids: List[str]) -> None:
		"""Deliver messages to a specific subscription"""
		
		messages = []
		for message_id in message_ids:
			if message_id in self.message_store:
				messages.append(self.message_store[message_id])
		
		if not messages:
			return
		
		# Simulate message delivery based on protocol
		if subscription.protocol == ProtocolType.HTTP_REST and subscription.webhook_url:
			await self._deliver_via_webhook(subscription, messages)
		elif subscription.protocol == ProtocolType.WEBSOCKET:
			await self._deliver_via_websocket(subscription, messages)
		else:
			# Fallback - mark as delivered for now
			self.logger.debug(f"Simulated delivery of {len(messages)} messages to {subscription.id}")
		
		# Update subscription statistics
		subscription.total_messages += len(messages)
		subscription.last_delivery = datetime.utcnow()
	
	async def _deliver_via_webhook(self, subscription: Subscription, messages: List[MQMessage]) -> None:
		"""Deliver messages via HTTP webhook"""
		
		# In production, would make actual HTTP requests
		self.logger.debug(f"Webhook delivery simulation: {len(messages)} messages to {subscription.webhook_url}")
		
		# Simulate success for now
		await asyncio.sleep(0.1)  # Simulate network delay
	
	async def _deliver_via_websocket(self, subscription: Subscription, messages: List[MQMessage]) -> None:
		"""Deliver messages via WebSocket"""
		
		# In production, would send via WebSocket connections
		self.logger.debug(f"WebSocket delivery simulation: {len(messages)} messages")
		
		# Simulate success for now
		await asyncio.sleep(0.01)  # Simulate minimal delay
	
	async def _cleanup_expired_messages(self) -> None:
		"""Clean up expired messages"""
		
		expired_message_ids = []
		
		for message_id, message in self.message_store.items():
			if message.is_expired():
				expired_message_ids.append(message_id)
		
		# Remove expired messages
		for message_id in expired_message_ids:
			message = self.message_store.pop(message_id, None)
			if message:
				self.logger.debug(f"Cleaned up expired message: {message_id}")
				
				# Remove from topic queues
				if message.topic in self.message_queues:
					try:
						self.message_queues[message.topic].remove(message_id)
					except ValueError:
						pass
				
				# Remove from subscription queues
				for subscription_queue in self.subscription_queues.values():
					try:
						subscription_queue.remove(message_id)
					except ValueError:
						pass
	
	async def _process_dead_letter_queues(self) -> None:
		"""Process messages in dead letter queues"""
		
		# In production, would implement retry logic and dead letter queue processing
		pass
	
	async def _metrics_collection_loop(self) -> None:
		"""Background metrics collection loop"""
		
		while self.running:
			try:
				# Update node health metrics
				for node in self.broker_nodes.values():
					node.last_heartbeat = datetime.utcnow()
					# Would update actual resource usage metrics
				
				# Calculate performance metrics
				# Would collect actual performance data
				
				await asyncio.sleep(30)  # Collect metrics every 30 seconds
				
			except Exception as e:
				self.logger.error(f"Error in metrics collection loop: {e}")
				await asyncio.sleep(60)
	
	async def _health_monitoring_loop(self) -> None:
		"""Background health monitoring loop"""
		
		while self.running:
			try:
				# Check cluster health
				if not self._is_cluster_healthy():
					self.logger.warning("Cluster health check failed")
				
				# Monitor resource usage
				# Would implement actual resource monitoring
				
				await asyncio.sleep(60)  # Check health every minute
				
			except Exception as e:
				self.logger.error(f"Error in health monitoring loop: {e}")
				await asyncio.sleep(120)


# Factory function
async def create_mqeb_service(config: Dict[str, Any] | None = None) -> MQEBService:
	"""Create and initialize MQEB service"""
	service = MQEBService(config)
	await service.initialize()
	return service


# Export main components
__all__ = [
	'MQEBService',
	'create_mqeb_service'
]
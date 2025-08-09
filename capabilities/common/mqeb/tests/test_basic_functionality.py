#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Basic Functionality Tests
Tests for core MQEB functionality

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import json
from datetime import datetime
from uuid_extensions import uuid7str

# Import MQEB components
from ..models import (
	MQMessage, TopicConfiguration, Subscription, MessageEvent, BrokerNode,
	MessagePriority, DeliveryMode, ProtocolType, MessageStatus
)
from ..service import MQEBService, create_mqeb_service


class TestMQEBModels:
	"""Test MQEB data models"""
	
	def test_message_model_creation(self):
		"""Test MQMessage model creation"""
		message = MQMessage(
			topic="test.topic",
			payload=b"Hello, World!",
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		assert message.topic == "test.topic"
		assert message.payload == b"Hello, World!"
		assert message.priority == MessagePriority.NORMAL
		assert message.delivery_mode == DeliveryMode.AT_LEAST_ONCE
		assert message.encrypted == False
		assert message.status == MessageStatus.PENDING
		assert isinstance(message.timestamp, datetime)
	
	def test_message_size_calculation(self):
		"""Test message size calculation"""
		message = MQMessage(
			topic="test.topic",
			payload=b"Hello, World!" * 100,
			headers={"content-type": "text/plain", "source": "test"},
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		size = message.size_bytes()
		assert size > 0
		assert size > len(message.payload)  # Should include headers and metadata
	
	def test_message_expiration_check(self):
		"""Test message expiration checking"""
		# Create message without expiration
		message = MQMessage(
			topic="test.topic",
			payload=b"test",
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		assert not message.is_expired()
		
		# Create message with past expiration
		past_expiration = datetime.utcnow().replace(year=2020)
		expired_message = MQMessage(
			topic="test.topic",
			payload=b"test",
			expiration=past_expiration,
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		assert expired_message.is_expired()
	
	def test_topic_configuration_model(self):
		"""Test TopicConfiguration model"""
		topic = TopicConfiguration(
			name="test.events",
			description="Test topic for events",
			partitions=5,
			replication_factor=3,
			retention_ms=604800000,  # 7 days
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		assert topic.name == "test.events"
		assert topic.partitions == 5
		assert topic.replication_factor == 3
		assert topic.encryption_required == True  # Default
		assert topic.schema_registry_enabled == True  # Default
	
	def test_subscription_model(self):
		"""Test Subscription model"""
		subscription = Subscription(
			name="test_subscription",
			topic_pattern="test.*",
			consumer_group="test_group",
			delivery_mode=DeliveryMode.EXACTLY_ONCE,
			protocol=ProtocolType.WEBSOCKET,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		assert subscription.name == "test_subscription"
		assert subscription.topic_pattern == "test.*"
		assert subscription.delivery_mode == DeliveryMode.EXACTLY_ONCE
		assert subscription.protocol == ProtocolType.WEBSOCKET
		assert subscription.enabled == True
		assert subscription.success_rate() == 1.0  # No failed messages initially
	
	def test_broker_node_model(self):
		"""Test BrokerNode model"""
		node = BrokerNode(
			name="test-broker-01",
			hostname="broker01.test.local",
			ip_address="192.168.1.10",
			port=8080,
			region="us-east-1",
			cluster_id="test-cluster"
		)
		
		assert node.name == "test-broker-01"
		assert node.hostname == "broker01.test.local"
		assert node.status == "active"
		assert node.is_healthy() == True  # Should be healthy with default values


class TestMQEBService:
	"""Test MQEB service functionality"""
	
	@pytest.fixture
	async def mqeb_service(self):
		"""Create MQEB service for testing"""
		service = MQEBService()
		await service.initialize()
		yield service
		await service.shutdown()
	
	@pytest.mark.asyncio
	async def test_service_initialization(self):
		"""Test service initialization"""
		service = MQEBService()
		assert not service.running
		
		await service.initialize()
		assert service.running
		assert len(service.broker_nodes) > 0
		assert len(service.topics) > 0  # Should have default topics
		
		await service.shutdown()
		assert not service.running
	
	@pytest.mark.asyncio
	async def test_topic_creation(self, mqeb_service):
		"""Test topic creation"""
		topic_config = TopicConfiguration(
			name="test.creation",
			description="Test topic creation",
			partitions=3,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		topic_name = await mqeb_service.create_topic(topic_config)
		
		assert topic_name == "test.creation"
		assert topic_name in mqeb_service.topics
		assert topic_name in mqeb_service.message_queues
		assert mqeb_service.metrics['topics_created'] > 0
	
	@pytest.mark.asyncio
	async def test_duplicate_topic_creation(self, mqeb_service):
		"""Test duplicate topic creation raises error"""
		topic_config = TopicConfiguration(
			name="test.duplicate",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		# Create first topic
		await mqeb_service.create_topic(topic_config)
		
		# Attempt to create duplicate should raise error
		with pytest.raises(ValueError, match="already exists"):
			await mqeb_service.create_topic(topic_config)
	
	@pytest.mark.asyncio
	async def test_message_publishing(self, mqeb_service):
		"""Test message publishing"""
		# Create a topic first
		topic_config = TopicConfiguration(
			name="test.publish",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		await mqeb_service.create_topic(topic_config)
		
		# Create and publish message
		message = MQMessage(
			topic="test.publish",
			payload=b"Test message payload",
			content_type="text/plain",
			priority=MessagePriority.HIGH,
			tenant_id="test_tenant",
			source_application="test_app",
			user_id="test_user"
		)
		
		message_id = await mqeb_service.publish_message(message)
		
		assert message_id == message.id
		assert message_id in mqeb_service.message_store
		assert message_id in mqeb_service.message_queues["test.publish"]
		assert mqeb_service.metrics['messages_published'] > 0
	
	@pytest.mark.asyncio
	async def test_message_publishing_to_nonexistent_topic(self, mqeb_service):
		"""Test publishing to non-existent topic raises error"""
		message = MQMessage(
			topic="nonexistent.topic",
			payload=b"Test message",
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		with pytest.raises(ValueError, match="does not exist"):
			await mqeb_service.publish_message(message)
	
	@pytest.mark.asyncio
	async def test_subscription_creation(self, mqeb_service):
		"""Test subscription creation"""
		subscription = Subscription(
			name="test_sub",
			topic_pattern="test.*",
			consumer_group="test_group",
			protocol=ProtocolType.HTTP_REST,
			webhook_url="http://localhost:8080/webhook",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		subscription_id = await mqeb_service.create_subscription(subscription)
		
		assert subscription_id == subscription.id
		assert subscription_id in mqeb_service.subscriptions
		assert subscription_id in mqeb_service.subscription_queues
		assert mqeb_service.metrics['subscriptions_created'] > 0
	
	@pytest.mark.asyncio
	async def test_message_routing_to_subscriptions(self, mqeb_service):
		"""Test message routing to matching subscriptions"""
		# Create topic
		topic_config = TopicConfiguration(
			name="test.routing",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		await mqeb_service.create_topic(topic_config)
		
		# Create subscription with matching pattern
		subscription = Subscription(
			name="routing_test_sub",
			topic_pattern="test.*",
			consumer_group="test_group",
			tenant_id="test_tenant",
			created_by="test_user"
		)
		subscription_id = await mqeb_service.create_subscription(subscription)
		
		# Publish message
		message = MQMessage(
			topic="test.routing",
			payload=b"Routing test message",
			tenant_id="test_tenant",
			source_application="test_app"
		)
		message_id = await mqeb_service.publish_message(message)
		
		# Message should be routed to subscription
		assert message_id in mqeb_service.subscription_queues[subscription_id]
	
	@pytest.mark.asyncio
	async def test_message_consumption(self, mqeb_service):
		"""Test message consumption from subscription"""
		# Create topic and subscription
		topic_config = TopicConfiguration(
			name="test.consume",
			tenant_id="test_tenant", 
			created_by="test_user"
		)
		await mqeb_service.create_topic(topic_config)
		
		subscription = Subscription(
			name="consume_test_sub",
			topic_pattern="test.consume",
			consumer_group="test_group",
			delivery_mode=DeliveryMode.AT_LEAST_ONCE,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		subscription_id = await mqeb_service.create_subscription(subscription)
		
		# Publish messages
		messages_published = []
		for i in range(5):
			message = MQMessage(
				topic="test.consume",
				payload=f"Test message {i}".encode(),
				tenant_id="test_tenant",
				source_application="test_app"
			)
			message_id = await mqeb_service.publish_message(message)
			messages_published.append(message_id)
		
		# Consume messages
		consumed_messages = await mqeb_service.consume_messages(subscription_id, max_messages=3)
		
		assert len(consumed_messages) == 3
		assert all(isinstance(msg, MQMessage) for msg in consumed_messages)
		assert all(msg.topic == "test.consume" for msg in consumed_messages)
		
		# Check that messages were removed from subscription queue
		remaining_in_queue = len(mqeb_service.subscription_queues[subscription_id])
		assert remaining_in_queue == 2  # 5 published - 3 consumed
	
	@pytest.mark.asyncio
	async def test_topic_statistics(self, mqeb_service):
		"""Test topic statistics retrieval"""
		# Create topic with messages
		topic_config = TopicConfiguration(
			name="test.stats",
			partitions=2,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		await mqeb_service.create_topic(topic_config)
		
		# Publish some messages
		for i in range(3):
			message = MQMessage(
				topic="test.stats",
				payload=f"Stats test message {i}".encode(),
				tenant_id="test_tenant",
				source_application="test_app"
			)
			await mqeb_service.publish_message(message)
		
		# Get topic stats
		stats = await mqeb_service.get_topic_stats("test.stats")
		
		assert stats['topic_name'] == "test.stats"
		assert stats['partitions'] == 2
		assert stats['total_messages'] == 3
		assert stats['total_size_bytes'] > 0
		assert 'created_at' in stats
	
	@pytest.mark.asyncio
	async def test_subscription_statistics(self, mqeb_service):
		"""Test subscription statistics retrieval"""
		# Create subscription
		subscription = Subscription(
			name="stats_test_sub",
			topic_pattern="stats.*",
			consumer_group="test_group",
			tenant_id="test_tenant", 
			created_by="test_user"
		)
		subscription_id = await mqeb_service.create_subscription(subscription)
		
		# Get subscription stats
		stats = await mqeb_service.get_subscription_stats(subscription_id)
		
		assert stats['subscription_id'] == subscription_id
		assert stats['name'] == "stats_test_sub"
		assert stats['topic_pattern'] == "stats.*"
		assert stats['enabled'] == True
		assert stats['success_rate'] == 1.0
		assert 'created_at' in stats
	
	@pytest.mark.asyncio
	async def test_cluster_statistics(self, mqeb_service):
		"""Test cluster statistics retrieval"""
		stats = await mqeb_service.get_cluster_stats()
		
		assert 'broker_nodes' in stats
		assert 'total_topics' in stats
		assert 'total_subscriptions' in stats
		assert 'metrics' in stats
		assert 'cluster_healthy' in stats
		assert stats['broker_nodes'] > 0


class TestMQEBPerformance:
	"""Performance tests for MQEB"""
	
	@pytest.mark.asyncio
	async def test_high_volume_message_publishing(self):
		"""Test high volume message publishing"""
		service = MQEBService()
		await service.initialize()
		
		try:
			# Create topic
			topic_config = TopicConfiguration(
				name="test.performance",
				partitions=5,
				tenant_id="test_tenant",
				created_by="test_user"
			)
			await service.create_topic(topic_config)
			
			# Publish many messages
			message_count = 1000
			start_time = datetime.utcnow()
			
			for i in range(message_count):
				message = MQMessage(
					topic="test.performance",
					payload=f"Performance test message {i}".encode(),
					tenant_id="test_tenant",
					source_application="test_app"
				)
				await service.publish_message(message)
			
			end_time = datetime.utcnow()
			duration_seconds = (end_time - start_time).total_seconds()
			
			# Calculate throughput
			throughput = message_count / duration_seconds
			
			assert throughput > 100  # Should handle at least 100 messages/second
			assert service.metrics['messages_published'] >= message_count
			
		finally:
			await service.shutdown()
	
	@pytest.mark.asyncio
	async def test_concurrent_topic_operations(self):
		"""Test concurrent topic operations"""
		service = MQEBService()
		await service.initialize()
		
		try:
			# Create multiple topics concurrently
			async def create_topic_with_messages(topic_index):
				topic_config = TopicConfiguration(
					name=f"test.concurrent.{topic_index}",
					tenant_id="test_tenant",
					created_by="test_user"
				)
				await service.create_topic(topic_config)
				
				# Publish messages to each topic
				for msg_index in range(10):
					message = MQMessage(
						topic=f"test.concurrent.{topic_index}",
						payload=f"Concurrent message {msg_index}".encode(),
						tenant_id="test_tenant",
						source_application="test_app"
					)
					await service.publish_message(message)
			
			# Run concurrent operations
			tasks = [create_topic_with_messages(i) for i in range(10)]
			await asyncio.gather(*tasks)
			
			# Verify results
			assert len(service.topics) >= 10  # At least 10 new topics + defaults
			assert service.metrics['messages_published'] >= 100  # 10 topics * 10 messages
			
		finally:
			await service.shutdown()


if __name__ == "__main__":
	# Run tests if script is executed directly
	pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Data Models
Pydantic models for MQEB core functionality with APG integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, validator, root_validator
from pydantic.types import conint, constr, conlist


class MessagePriority(str, Enum):
	"""Message priority levels"""
	LOW = "low"
	NORMAL = "normal" 
	HIGH = "high"
	CRITICAL = "critical"


class DeliveryMode(str, Enum):
	"""Message delivery guarantees"""
	AT_MOST_ONCE = "at_most_once"		# Fire and forget
	AT_LEAST_ONCE = "at_least_once"		# Deliver at least once
	EXACTLY_ONCE = "exactly_once"		# Deliver exactly once


class CompressionType(str, Enum):
	"""Message compression algorithms"""
	NONE = "none"
	GZIP = "gzip"
	SNAPPY = "snappy"
	LZ4 = "lz4"
	ZSTD = "zstd"


class ProtocolType(str, Enum):
	"""Supported messaging protocols"""
	MQTT = "mqtt"
	AMQP = "amqp"
	KAFKA = "kafka"
	WEBSOCKET = "websocket"
	GRPC = "grpc"
	HTTP_REST = "http_rest"


class MessageStatus(str, Enum):
	"""Message processing status"""
	PENDING = "pending"
	PROCESSING = "processing"
	DELIVERED = "delivered"
	FAILED = "failed"
	EXPIRED = "expired"
	DEAD_LETTER = "dead_letter"


class TopicType(str, Enum):
	"""Topic configuration types"""
	STANDARD = "standard"		# Normal pub/sub
	FIFO = "fifo"			   # First-in-first-out ordering
	FANOUT = "fanout"		   # Broadcast to all subscribers
	DIRECT = "direct"		   # Direct routing
	TOPIC = "topic"			 # Pattern-based routing


class EncryptionMode(str, Enum):
	"""Message encryption modes"""
	NONE = "none"
	TRANSPORT = "transport"		 # TLS encryption only
	MESSAGE = "message"			 # Message-level encryption
	END_TO_END = "end_to_end"	   # Full end-to-end encryption


class RetryStrategy(str, Enum):
	"""Message retry strategies"""
	NONE = "none"
	LINEAR = "linear"
	EXPONENTIAL = "exponential"
	CUSTOM = "custom"


class MQMessage(BaseModel):
	"""Core message model for MQEB"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	# Message identification
	id: str = Field(default_factory=uuid7str, description="Unique message identifier")
	correlation_id: Optional[str] = Field(None, description="Correlation ID for request/response")
	reply_to: Optional[str] = Field(None, description="Reply-to topic for responses")
	
	# Routing information
	topic: constr(min_length=1, max_length=255) = Field(..., description="Message topic")
	partition_key: Optional[str] = Field(None, description="Partition key for routing")
	routing_key: Optional[str] = Field(None, description="Routing key for exchanges")
	
	# Message content
	payload: bytes = Field(..., description="Message payload data")
	content_type: str = Field(default="application/octet-stream", description="Payload content type")
	content_encoding: Optional[str] = Field(None, description="Payload encoding")
	
	# Message metadata
	headers: Dict[str, str] = Field(default_factory=dict, description="Message headers")
	properties: Dict[str, Any] = Field(default_factory=dict, description="Message properties")
	
	# Delivery configuration
	priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority")
	delivery_mode: DeliveryMode = Field(default=DeliveryMode.AT_LEAST_ONCE, description="Delivery guarantee")
	
	# Timing
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message creation time")
	expiration: Optional[datetime] = Field(None, description="Message expiration time")
	delivery_delay: Optional[int] = Field(None, ge=0, description="Delivery delay in milliseconds")
	
	# Security and compliance
	encrypted: bool = Field(default=False, description="Whether message is encrypted")
	encryption_key_id: Optional[str] = Field(None, description="Encryption key identifier")
	signature: Optional[str] = Field(None, description="Message signature for integrity")
	
	# APG integration
	tenant_id: str = Field(..., description="APG tenant identifier")
	source_application: str = Field(..., description="Source application identifier")
	user_id: Optional[str] = Field(None, description="User who sent the message")
	trace_id: Optional[str] = Field(None, description="Distributed trace identifier")
	
	# Message lifecycle
	status: MessageStatus = Field(default=MessageStatus.PENDING, description="Message status")
	retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
	max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
	
	# Schema information
	schema_id: Optional[str] = Field(None, description="Message schema identifier")
	schema_version: Optional[str] = Field(None, description="Message schema version")
	
	@validator('payload')
	def validate_payload_size(cls, v):
		"""Validate message payload size"""
		if len(v) > 100 * 1024 * 1024:  # 100MB limit
			raise ValueError("Message payload exceeds maximum size of 100MB")
		return v
	
	@validator('expiration')
	def validate_expiration(cls, v, values):
		"""Validate expiration is in the future"""
		if v and v <= datetime.utcnow():
			raise ValueError("Message expiration must be in the future")
		return v
	
	def is_expired(self) -> bool:
		"""Check if message has expired"""
		return self.expiration and datetime.utcnow() > self.expiration
	
	def size_bytes(self) -> int:
		"""Calculate total message size in bytes"""
		header_size = sum(len(k.encode()) + len(v.encode()) for k, v in self.headers.items())
		return len(self.payload) + header_size + 1024  # Approximate metadata overhead


class TopicConfiguration(BaseModel):
	"""Topic configuration model"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	# Topic identification
	name: constr(min_length=1, max_length=255) = Field(..., description="Topic name")
	display_name: Optional[str] = Field(None, description="Human-readable topic name")
	description: Optional[str] = Field(None, description="Topic description")
	
	# Topic type and behavior
	topic_type: TopicType = Field(default=TopicType.STANDARD, description="Topic type")
	fifo_enabled: bool = Field(default=False, description="Enable FIFO ordering")
	
	# Partitioning configuration
	partitions: conint(ge=1, le=10000) = Field(default=1, description="Number of partitions")
	auto_partition: bool = Field(default=True, description="Enable automatic partitioning")
	partition_strategy: str = Field(default="hash", description="Partitioning strategy")
	
	# Replication and durability
	replication_factor: conint(ge=1, le=10) = Field(default=3, description="Replication factor")
	min_in_sync_replicas: conint(ge=1) = Field(default=2, description="Minimum in-sync replicas")
	
	# Retention configuration
	retention_ms: conint(ge=1000) = Field(default=604800000, description="Retention time in milliseconds")  # 7 days
	retention_bytes: Optional[conint(ge=1)] = Field(None, description="Retention size in bytes")
	cleanup_policy: str = Field(default="delete", description="Cleanup policy (delete/compact)")
	
	# Message configuration
	max_message_size: conint(ge=1, le=104857600) = Field(default=1048576, description="Maximum message size")  # 1MB
	compression_type: CompressionType = Field(default=CompressionType.SNAPPY, description="Compression type")
	
	# Security configuration
	encryption_required: bool = Field(default=True, description="Require message encryption")
	encryption_mode: EncryptionMode = Field(default=EncryptionMode.MESSAGE, description="Encryption mode")
	signing_required: bool = Field(default=False, description="Require message signing")
	
	# Schema configuration
	schema_registry_enabled: bool = Field(default=True, description="Enable schema registry")
	schema_validation_enabled: bool = Field(default=False, description="Enable schema validation")
	schema_evolution_mode: str = Field(default="backward_compatible", description="Schema evolution mode")
	
	# Dead letter queue configuration
	dead_letter_queue: Optional[str] = Field(None, description="Dead letter queue topic")
	max_delivery_attempts: conint(ge=1) = Field(default=5, description="Maximum delivery attempts")
	
	# APG integration
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="User who created the topic")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	# Access control
	public_read: bool = Field(default=False, description="Allow public read access")
	public_write: bool = Field(default=False, description="Allow public write access")
	allowed_producers: List[str] = Field(default_factory=list, description="Allowed producer applications")
	allowed_consumers: List[str] = Field(default_factory=list, description="Allowed consumer applications")
	
	# Performance tuning
	batch_size: conint(ge=1) = Field(default=100, description="Default batch size for consumers")
	buffer_size: conint(ge=1024) = Field(default=65536, description="Buffer size in bytes")
	flush_interval_ms: conint(ge=1) = Field(default=1000, description="Flush interval in milliseconds")
	
	@validator('min_in_sync_replicas')
	def validate_min_in_sync_replicas(cls, v, values):
		"""Validate min in sync replicas doesn't exceed replication factor"""
		if 'replication_factor' in values and v > values['replication_factor']:
			raise ValueError("min_in_sync_replicas cannot exceed replication_factor")
		return v


class MessageFilter(BaseModel):
	"""Message filtering configuration"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	# Header-based filtering
	header_filters: Dict[str, str] = Field(default_factory=dict, description="Header-based filters")
	property_filters: Dict[str, Any] = Field(default_factory=dict, description="Property-based filters")
	
	# Content-based filtering
	content_filters: List[str] = Field(default_factory=list, description="Content-based filter expressions")
	content_type_filter: Optional[str] = Field(None, description="Content type filter")
	
	# Source filtering
	source_applications: List[str] = Field(default_factory=list, description="Allowed source applications")
	user_filters: List[str] = Field(default_factory=list, description="User-based filters")
	
	# Time-based filtering
	time_range_start: Optional[datetime] = Field(None, description="Time range start")
	time_range_end: Optional[datetime] = Field(None, description="Time range end")
	
	# Priority filtering
	min_priority: Optional[MessagePriority] = Field(None, description="Minimum message priority")
	max_priority: Optional[MessagePriority] = Field(None, description="Maximum message priority")
	
	def matches(self, message: MQMessage) -> bool:
		"""Check if message matches the filter criteria"""
		# Header filtering
		for key, value in self.header_filters.items():
			if key not in message.headers or message.headers[key] != value:
				return False
		
		# Property filtering
		for key, value in self.property_filters.items():
			if key not in message.properties or message.properties[key] != value:
				return False
		
		# Source application filtering
		if self.source_applications and message.source_application not in self.source_applications:
			return False
		
		# User filtering
		if self.user_filters and message.user_id not in self.user_filters:
			return False
		
		# Time range filtering
		if self.time_range_start and message.timestamp < self.time_range_start:
			return False
		if self.time_range_end and message.timestamp > self.time_range_end:
			return False
		
		# Priority filtering
		if self.min_priority and message.priority.value < self.min_priority.value:
			return False
		if self.max_priority and message.priority.value > self.max_priority.value:
			return False
		
		return True


class RetryPolicy(BaseModel):
	"""Message retry policy configuration"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	strategy: RetryStrategy = Field(default=RetryStrategy.EXPONENTIAL, description="Retry strategy")
	max_attempts: conint(ge=0, le=100) = Field(default=3, description="Maximum retry attempts")
	initial_delay_ms: conint(ge=100) = Field(default=1000, description="Initial retry delay")
	max_delay_ms: conint(ge=1000) = Field(default=300000, description="Maximum retry delay")  # 5 minutes
	backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0, description="Backoff multiplier")
	
	# Jitter configuration
	jitter_enabled: bool = Field(default=True, description="Enable jitter to avoid thundering herd")
	jitter_max_ms: conint(ge=0) = Field(default=1000, description="Maximum jitter in milliseconds")
	
	# Retry conditions
	retry_on_timeout: bool = Field(default=True, description="Retry on timeout errors")
	retry_on_connection_error: bool = Field(default=True, description="Retry on connection errors")
	retry_on_server_error: bool = Field(default=True, description="Retry on server errors")
	retry_on_client_error: bool = Field(default=False, description="Retry on client errors")
	
	# Custom retry conditions
	retryable_error_codes: List[str] = Field(default_factory=list, description="Custom retryable error codes")
	non_retryable_error_codes: List[str] = Field(default_factory=list, description="Custom non-retryable error codes")
	
	def calculate_delay(self, attempt: int) -> int:
		"""Calculate delay for retry attempt"""
		if self.strategy == RetryStrategy.LINEAR:
			delay = self.initial_delay_ms * attempt
		elif self.strategy == RetryStrategy.EXPONENTIAL:
			delay = self.initial_delay_ms * (self.backoff_multiplier ** (attempt - 1))
		else:
			delay = self.initial_delay_ms
		
		# Apply maximum delay limit
		delay = min(delay, self.max_delay_ms)
		
		# Apply jitter if enabled
		if self.jitter_enabled:
			import random
			jitter = random.randint(0, self.jitter_max_ms)
			delay += jitter
		
		return int(delay)


class Subscription(BaseModel):
	"""Message subscription configuration"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	# Subscription identification
	id: str = Field(default_factory=uuid7str, description="Subscription identifier")
	name: constr(min_length=1, max_length=255) = Field(..., description="Subscription name")
	description: Optional[str] = Field(None, description="Subscription description")
	
	# Topic subscription
	topic_pattern: constr(min_length=1, max_length=1000) = Field(..., description="Topic pattern to subscribe to")
	consumer_group: constr(min_length=1, max_length=255) = Field(..., description="Consumer group identifier")
	
	# Delivery configuration
	delivery_mode: DeliveryMode = Field(default=DeliveryMode.AT_LEAST_ONCE, description="Delivery guarantee")
	protocol: ProtocolType = Field(default=ProtocolType.HTTP_REST, description="Delivery protocol")
	
	# Message filtering
	message_filter: Optional[MessageFilter] = Field(None, description="Message filter configuration")
	
	# Batch configuration
	batch_enabled: bool = Field(default=False, description="Enable batch message delivery")
	batch_size: conint(ge=1, le=10000) = Field(default=1, description="Batch size")
	max_wait_time_ms: conint(ge=100, le=300000) = Field(default=1000, description="Maximum wait time for batching")
	
	# Retry configuration
	retry_policy: RetryPolicy = Field(default_factory=RetryPolicy, description="Retry policy")
	dead_letter_queue: Optional[str] = Field(None, description="Dead letter queue topic")
	
	# Delivery endpoints
	webhook_url: Optional[str] = Field(None, description="Webhook URL for HTTP delivery")
	queue_name: Optional[str] = Field(None, description="Queue name for queue-based delivery")
	callback_config: Optional[Dict[str, Any]] = Field(None, description="Callback configuration")
	
	# Security configuration
	authentication_enabled: bool = Field(default=True, description="Enable authentication")
	authentication_config: Dict[str, Any] = Field(default_factory=dict, description="Authentication configuration")
	encryption_required: bool = Field(default=True, description="Require encrypted delivery")
	
	# APG integration
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="User who created the subscription")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	# Subscription state
	enabled: bool = Field(default=True, description="Subscription enabled state")
	paused: bool = Field(default=False, description="Subscription paused state")
	last_delivery: Optional[datetime] = Field(None, description="Last successful delivery time")
	total_messages: int = Field(default=0, ge=0, description="Total messages delivered")
	failed_messages: int = Field(default=0, ge=0, description="Total failed message deliveries")
	
	# Performance configuration
	max_concurrent_deliveries: conint(ge=1, le=1000) = Field(default=10, description="Maximum concurrent deliveries")
	delivery_timeout_ms: conint(ge=1000, le=300000) = Field(default=30000, description="Delivery timeout")
	
	def success_rate(self) -> float:
		"""Calculate delivery success rate"""
		if self.total_messages == 0:
			return 1.0
		return (self.total_messages - self.failed_messages) / self.total_messages


class MessageEvent(BaseModel):
	"""Message lifecycle event for auditing"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	# Event identification
	id: str = Field(default_factory=uuid7str, description="Event identifier")
	message_id: str = Field(..., description="Related message identifier")
	subscription_id: Optional[str] = Field(None, description="Related subscription identifier")
	
	# Event details
	event_type: str = Field(..., description="Event type (published, delivered, failed, etc.)")
	event_time: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
	status: str = Field(..., description="Event status")
	
	# Context information
	broker_node_id: Optional[str] = Field(None, description="Broker node that processed the event")
	protocol: Optional[ProtocolType] = Field(None, description="Protocol used")
	endpoint: Optional[str] = Field(None, description="Delivery endpoint")
	
	# Error information
	error_code: Optional[str] = Field(None, description="Error code if applicable")
	error_message: Optional[str] = Field(None, description="Error message if applicable")
	retry_attempt: Optional[int] = Field(None, ge=0, description="Retry attempt number")
	
	# Performance metrics
	processing_time_ms: Optional[int] = Field(None, ge=0, description="Processing time in milliseconds")
	queue_time_ms: Optional[int] = Field(None, ge=0, description="Time spent in queue")
	delivery_time_ms: Optional[int] = Field(None, ge=0, description="Delivery time in milliseconds")
	
	# APG integration
	tenant_id: str = Field(..., description="APG tenant identifier")
	user_id: Optional[str] = Field(None, description="User associated with the event")
	trace_id: Optional[str] = Field(None, description="Distributed trace identifier")
	
	# Additional context
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")


class BrokerNode(BaseModel):
	"""Message broker node configuration"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	# Node identification
	id: str = Field(default_factory=uuid7str, description="Node identifier")
	name: constr(min_length=1, max_length=255) = Field(..., description="Node name")
	hostname: str = Field(..., description="Node hostname")
	ip_address: str = Field(..., description="Node IP address")
	port: conint(ge=1, le=65535) = Field(default=8080, description="Node port")
	
	# Node configuration
	node_type: str = Field(default="broker", description="Node type (broker, gateway, edge)")
	region: str = Field(..., description="Geographic region")
	availability_zone: Optional[str] = Field(None, description="Availability zone")
	data_center: Optional[str] = Field(None, description="Data center identifier")
	
	# Resource configuration
	max_connections: conint(ge=1) = Field(default=10000, description="Maximum concurrent connections")
	max_topics: conint(ge=1) = Field(default=10000, description="Maximum topics per node")
	max_partitions: conint(ge=1) = Field(default=100000, description="Maximum partitions per node")
	memory_limit_mb: conint(ge=512) = Field(default=4096, description="Memory limit in MB")
	disk_limit_gb: conint(ge=1) = Field(default=100, description="Disk limit in GB")
	
	# Network configuration
	protocols_enabled: List[ProtocolType] = Field(
		default=[ProtocolType.HTTP_REST, ProtocolType.WEBSOCKET],
		description="Enabled protocols"
	)
	ssl_enabled: bool = Field(default=True, description="SSL/TLS enabled")
	ssl_certificate_path: Optional[str] = Field(None, description="SSL certificate path")
	
	# Cluster configuration
	cluster_id: str = Field(..., description="Cluster identifier")
	leader: bool = Field(default=False, description="Whether this node is cluster leader")
	cluster_members: List[str] = Field(default_factory=list, description="Other cluster member node IDs")
	
	# Node status
	status: str = Field(default="active", description="Node status (active, inactive, maintenance)")
	last_heartbeat: datetime = Field(default_factory=datetime.utcnow, description="Last heartbeat time")
	start_time: datetime = Field(default_factory=datetime.utcnow, description="Node start time")
	
	# Performance metrics
	cpu_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="CPU usage percentage")
	memory_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="Memory usage percentage")
	disk_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="Disk usage percentage")
	network_io_mbps: float = Field(default=0.0, ge=0.0, description="Network I/O in Mbps")
	
	# Message statistics
	messages_processed: int = Field(default=0, ge=0, description="Total messages processed")
	messages_per_second: float = Field(default=0.0, ge=0.0, description="Current messages per second")
	bytes_processed: int = Field(default=0, ge=0, description="Total bytes processed")
	active_connections: int = Field(default=0, ge=0, description="Current active connections")
	
	# APG integration
	tenant_id: Optional[str] = Field(None, description="APG tenant identifier (for tenant-specific nodes)")
	
	def is_healthy(self) -> bool:
		"""Check if node is healthy"""
		now = datetime.utcnow()
		heartbeat_age = (now - self.last_heartbeat).total_seconds()
		
		return (
			self.status == "active" and
			heartbeat_age < 300 and  # 5 minutes
			self.cpu_usage < 90 and
			self.memory_usage < 90 and
			self.disk_usage < 90
		)


# Export main components
__all__ = [
	'MQMessage', 'TopicConfiguration', 'MessageFilter', 'RetryPolicy', 
	'Subscription', 'MessageEvent', 'BrokerNode',
	'MessagePriority', 'DeliveryMode', 'CompressionType', 'ProtocolType',
	'MessageStatus', 'TopicType', 'EncryptionMode', 'RetryStrategy'
]
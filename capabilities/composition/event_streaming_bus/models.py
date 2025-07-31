"""
APG Event Streaming Bus - Data Models

Comprehensive data models for event streaming, stream processing, and event sourcing
with Apache Kafka integration and enterprise-grade event management.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from sqlalchemy import (
	Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
	ForeignKey, Index, UniqueConstraint, CheckConstraint, BigInteger,
	LargeBinary, SmallInteger, Numeric
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID as PostgreSQL_UUID, JSONB, ARRAY

from pydantic import BaseModel, ConfigDict, Field, validator, root_validator
from pydantic.types import UUID4
from uuid_extensions import uuid7str

# =============================================================================
# Database Base and Configuration
# =============================================================================

Base = declarative_base()

# Common model configuration
model_config = ConfigDict(
    extra='forbid',
    validate_by_name=True,
    validate_by_alias=True,
    str_strip_whitespace=True,
    use_enum_values=True
)

# =============================================================================
# Enums and Constants
# =============================================================================

class EventStatus(str, Enum):
	"""Event processing status"""
	PENDING = "pending"
	PROCESSING = "processing"
	PROCESSED = "processed"
	PUBLISHED = "published"
	CONSUMED = "consumed"
	FAILED = "failed"
	RETRYING = "retrying"
	DEAD_LETTER = "dead_letter"
	ARCHIVED = "archived"

class EventPriority(str, Enum):
	"""Event priority levels"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	CRITICAL = "critical"
	EMERGENCY = "emergency"

class StreamStatus(str, Enum):
	"""Stream operational status"""
	ACTIVE = "active"
	PAUSED = "paused"
	STOPPED = "stopped"
	ERROR = "error"
	MIGRATING = "migrating"
	ARCHIVED = "archived"

class SubscriptionStatus(str, Enum):
	"""Subscription status"""
	ACTIVE = "active"
	PAUSED = "paused"
	CANCELLED = "cancelled"
	ERROR = "error"

class ConsumerStatus(str, Enum):
	"""Consumer group status"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	REBALANCING = "rebalancing"
	ERROR = "error"
	DEAD = "dead"

class ProcessorType(str, Enum):
	"""Stream processor types"""
	FILTER = "filter"
	MAP = "map"
	AGGREGATE = "aggregate"
	JOIN = "join"
	WINDOW = "window"
	COMPLEX_EVENT = "complex_event"
	ENRICHMENT = "enrichment"
	VALIDATION = "validation"

class EventType(str, Enum):
	"""Event type categories"""
	DOMAIN_EVENT = "domain_event"
	INTEGRATION_EVENT = "integration_event"
	NOTIFICATION_EVENT = "notification_event"
	SYSTEM_EVENT = "system_event"
	AUDIT_EVENT = "audit_event"

class DeliveryMode(str, Enum):
	"""Message delivery guarantees"""
	AT_MOST_ONCE = "at_most_once"
	AT_LEAST_ONCE = "at_least_once"
	EXACTLY_ONCE = "exactly_once"

class CompressionType(str, Enum):
	"""Compression algorithms"""
	NONE = "none"
	GZIP = "gzip"
	SNAPPY = "snappy"
	LZ4 = "lz4"
	ZSTD = "zstd"

class SerializationFormat(str, Enum):
	"""Event serialization formats"""
	JSON = "json"
	AVRO = "avro"
	PROTOBUF = "protobuf"
	MSGPACK = "msgpack"
	BINARY = "binary"

# =============================================================================
# Core Event Model
# =============================================================================

class ESEvent(Base):
	"""Core event entity for event streaming bus"""
	__tablename__ = "es_events"
	
	# Primary identification
	event_id = Column(String(30), primary_key=True, default=uuid7str)
	event_type = Column(String(200), nullable=False, index=True)
	event_version = Column(String(20), nullable=False, default="1.0")
	
	# Source and targeting
	source_capability = Column(String(100), nullable=False, index=True)
	target_capability = Column(String(100), nullable=True, index=True)
	aggregate_id = Column(String(50), nullable=False, index=True)
	aggregate_type = Column(String(100), nullable=False, index=True)
	
	# Event ordering and correlation
	sequence_number = Column(BigInteger, nullable=False, index=True)
	correlation_id = Column(String(50), nullable=True, index=True)
	causation_id = Column(String(30), nullable=True, index=True)
	
	# Timestamps
	event_timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	ingestion_timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	processed_timestamp = Column(DateTime(timezone=True), nullable=True)
	
	# Multi-tenancy and security
	tenant_id = Column(String(50), nullable=False, index=True)
	user_id = Column(String(50), nullable=True, index=True)
	session_id = Column(String(50), nullable=True)
	
	# Event status and processing
	status = Column(String(20), nullable=False, default=EventStatus.PENDING, index=True)
	priority = Column(String(20), nullable=False, default=EventPriority.NORMAL, index=True)
	retry_count = Column(Integer, nullable=False, default=0)
	max_retries = Column(Integer, nullable=False, default=3)
	
	# Content and metadata
	payload = Column(JSONB, nullable=False)
	metadata = Column(JSONB, nullable=False, default=dict)
	headers = Column(JSONB, nullable=False, default=dict)
	
	# Schema and validation
	schema_id = Column(String(50), nullable=True, index=True)
	schema_version = Column(String(20), nullable=False, default="1.0")
	content_type = Column(String(100), nullable=False, default="application/json")
	
	# Serialization and compression
	serialization_format = Column(String(20), nullable=False, default=SerializationFormat.JSON)
	compression_type = Column(String(20), nullable=False, default=CompressionType.NONE)
	original_size = Column(Integer, nullable=True)
	compressed_size = Column(Integer, nullable=True)
	
	# Processing metrics
	processing_duration_ms = Column(Integer, nullable=True)
	bytes_processed = Column(BigInteger, nullable=True)
	
	# Error handling
	error_message = Column(Text, nullable=True)
	error_code = Column(String(50), nullable=True)
	error_details = Column(JSONB, nullable=True)
	
	# Stream assignment (keep compatibility with existing code)
	stream_id = Column(String(100), nullable=False, index=True)
	partition_key = Column(String(200), nullable=True)
	offset_position = Column(BigInteger, nullable=True)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	created_by = Column(String(50), nullable=False)
	updated_by = Column(String(50), nullable=False)
	
	# Relationships
	stream = relationship("ESStream", back_populates="events")
	audit_logs = relationship("ESAuditLog", back_populates="event")
	stream_assignments = relationship("ESStreamAssignment", back_populates="event", cascade="all, delete-orphan")
	processing_history = relationship("ESEventProcessingHistory", back_populates="event", cascade="all, delete-orphan")
	
	# Indexes and constraints
	__table_args__ = (
		Index('idx_es_events_tenant_type_timestamp', 'tenant_id', 'event_type', 'event_timestamp'),
		Index('idx_es_events_aggregate', 'aggregate_type', 'aggregate_id', 'sequence_number'),
		Index('idx_es_events_correlation', 'correlation_id', 'event_timestamp'),
		Index('idx_es_events_status_priority', 'status', 'priority', 'event_timestamp'),
		Index('idx_es_events_source_target', 'source_capability', 'target_capability'),
		Index('idx_es_events_stream_offset', 'stream_id', 'offset_position'),
		CheckConstraint('retry_count >= 0', name='ck_es_events_retry_count_positive'),
		CheckConstraint('max_retries >= 0', name='ck_es_events_max_retries_positive'),
		CheckConstraint('sequence_number > 0', name='ck_es_events_sequence_positive'),
		CheckConstraint('original_size >= 0', name='ck_es_events_original_size_positive'),
		CheckConstraint('compressed_size >= 0', name='ck_es_events_compressed_size_positive'),
	)
	
	@validates('status')
	def validate_status(self, key, value):
		if value not in [status.value for status in EventStatus]:
			raise ValueError(f"Invalid event status: {value}")
		return value
	
	@validates('priority')
	def validate_priority(self, key, value):
		if value not in [priority.value for priority in EventPriority]:
			raise ValueError(f"Invalid event priority: {value}")
		return value
	
	@validates('event_type')
	def validate_event_type(self, key, value):
		if not value or len(value.strip()) == 0:
			raise ValueError("Event type cannot be empty")
		return value.strip()
	
	@validates('payload')
	def validate_payload(self, key, value):
		if not isinstance(value, dict):
			raise ValueError("Payload must be a dictionary")
		return value
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert event to dictionary representation"""
		return {
			"event_id": self.event_id,
			"event_type": self.event_type,
			"event_version": self.event_version,
			"source_capability": self.source_capability,
			"target_capability": self.target_capability,
			"aggregate_id": self.aggregate_id,
			"aggregate_type": self.aggregate_type,
			"sequence_number": self.sequence_number,
			"correlation_id": self.correlation_id,
			"causation_id": self.causation_id,
			"event_timestamp": self.event_timestamp.isoformat() if self.event_timestamp else None,
			"tenant_id": self.tenant_id,
			"user_id": self.user_id,
			"status": self.status,
			"priority": self.priority,
			"payload": self.payload,
			"metadata": self.metadata,
			"headers": self.headers,
			"schema_version": self.schema_version
		}
	
	def __repr__(self):
		return f"<ESEvent(id={self.event_id}, type={self.event_type}, aggregate={self.aggregate_id})>"

# =============================================================================
# Stream Configuration Model
# =============================================================================

class ESStream(Base):
    """Event stream configuration and metadata."""
    
    __tablename__ = "es_streams"
    
    # Primary identification
    stream_id = Column(String(100), primary_key=True, default=lambda: f"str_{uuid7str()}")
    stream_name = Column(String(200), nullable=False, unique=True)
    stream_description = Column(Text, nullable=True)
    
    # Kafka topic configuration
    topic_name = Column(String(200), nullable=False, unique=True)
    partitions = Column(Integer, nullable=False, default=3)
    replication_factor = Column(Integer, nullable=False, default=3)
    
    # Retention policies
    retention_time_ms = Column(BigInteger, nullable=False, default=604800000)  # 7 days
    retention_size_bytes = Column(BigInteger, nullable=True)
    cleanup_policy = Column(String(20), nullable=False, default="delete")  # delete, compact
    
    # Compression and serialization
    compression_type = Column(String(20), nullable=False, default=CompressionType.SNAPPY.value)
    default_serialization = Column(String(20), nullable=False, default=SerializationFormat.JSON.value)
    
    # Stream categorization
    event_category = Column(String(100), nullable=False, default=EventType.DOMAIN_EVENT.value)
    source_capability = Column(String(100), nullable=False)
    
    # Configuration settings
    config_settings = Column(JSONB, nullable=False, default={})
    
    # Status and monitoring
    status = Column(String(20), nullable=False, default=StreamStatus.ACTIVE.value)
    
    # Multi-tenancy
    tenant_id = Column(String(100), nullable=False, index=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100), nullable=False)
    
    # Relationships
    events = relationship("ESEvent", back_populates="stream")
    subscriptions = relationship("ESSubscription", back_populates="stream")
    metrics = relationship("ESMetrics", back_populates="stream")
    assignments = relationship("ESStreamAssignment", back_populates="stream", cascade="all, delete-orphan")
    consumer_groups = relationship("ESConsumerGroup", back_populates="stream", cascade="all, delete-orphan")
    processors = relationship("ESStreamProcessor", back_populates="stream", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_es_streams_tenant', 'tenant_id'),
        Index('idx_es_streams_capability', 'source_capability'),
        Index('idx_es_streams_status', 'status'),
        CheckConstraint('partitions > 0', name='check_partitions_positive'),
        CheckConstraint('replication_factor > 0', name='check_replication_positive'),
        CheckConstraint('retention_time_ms > 0', name='check_retention_time_positive')
    )
    
    @validates('stream_name')
    def validate_stream_name(self, key, value):
        if not value or len(value.strip()) == 0:
            raise ValueError("Stream name cannot be empty")
        # Kafka topic naming rules
        if not all(c.isalnum() or c in '._-' for c in value):
            raise ValueError("Stream name can only contain alphanumeric characters, dots, underscores, and hyphens")
        return value.strip()
    
    def __repr__(self):
        return f"<ESStream(id={self.stream_id}, name={self.stream_name}, topic={self.topic_name})>"

# =============================================================================
# Subscription Model
# =============================================================================

class ESSubscription(Base):
    """Event subscription configuration for consumers."""
    
    __tablename__ = "es_subscriptions"
    
    # Primary identification
    subscription_id = Column(String(100), primary_key=True, default=lambda: f"sub_{uuid7str()}")
    subscription_name = Column(String(200), nullable=False)
    subscription_description = Column(Text, nullable=True)
    
    # Stream relationship
    stream_id = Column(String(100), ForeignKey("es_streams.stream_id"), nullable=False, index=True)
    
    # Consumer configuration
    consumer_group_id = Column(String(100), nullable=False, index=True)
    consumer_name = Column(String(200), nullable=False)
    
    # Event filtering
    event_type_patterns = Column(JSONB, nullable=False, default=[])  # List of patterns like "user.*"
    filter_criteria = Column(JSONB, nullable=False, default={})  # Additional filter conditions
    
    # Delivery configuration
    delivery_mode = Column(String(20), nullable=False, default=DeliveryMode.AT_LEAST_ONCE.value)
    batch_size = Column(Integer, nullable=False, default=100)
    max_wait_time_ms = Column(Integer, nullable=False, default=1000)
    
    # Consumer position
    start_position = Column(String(20), nullable=False, default="latest")  # earliest, latest, specific_offset
    specific_offset = Column(BigInteger, nullable=True)
    
    # Retry and error handling
    retry_policy = Column(JSONB, nullable=False, default={
        "max_retries": 3,
        "retry_delay_ms": 1000,
        "backoff_multiplier": 2.0,
        "max_delay_ms": 60000
    })
    dead_letter_enabled = Column(Boolean, nullable=False, default=True)
    dead_letter_topic = Column(String(200), nullable=True)
    
    # Webhook delivery (if applicable)
    webhook_url = Column(String(500), nullable=True)
    webhook_headers = Column(JSONB, nullable=True, default={})
    webhook_timeout_ms = Column(Integer, nullable=True, default=30000)
    
    # Status and monitoring
    status = Column(String(20), nullable=False, default=SubscriptionStatus.ACTIVE.value)
    last_consumed_offset = Column(BigInteger, nullable=True)
    last_consumed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Multi-tenancy
    tenant_id = Column(String(100), nullable=False, index=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100), nullable=False)
    
    # Relationships
    stream = relationship("ESStream", back_populates="subscriptions")
    consumer_group = relationship("ESConsumerGroup", back_populates="subscriptions")
    
    # Indexes
    __table_args__ = (
        Index('idx_es_subscriptions_tenant', 'tenant_id'),
        Index('idx_es_subscriptions_consumer_group', 'consumer_group_id'),
        Index('idx_es_subscriptions_status', 'status'),
        Index('idx_es_subscriptions_stream_status', 'stream_id', 'status'),
        UniqueConstraint('subscription_name', 'tenant_id', name='uq_subscription_name_tenant'),
        CheckConstraint('batch_size > 0', name='check_batch_size_positive'),
        CheckConstraint('max_wait_time_ms > 0', name='check_wait_time_positive')
    )
    
    @validates('event_type_patterns')
    def validate_event_patterns(self, key, value):
        if not isinstance(value, list):
            raise ValueError("Event type patterns must be a list")
        return value
    
    def __repr__(self):
        return f"<ESSubscription(id={self.subscription_id}, name={self.subscription_name}, group={self.consumer_group_id})>"

# =============================================================================
# Consumer Group Model
# =============================================================================

class ESConsumerGroup(Base):
    """Consumer group configuration and state management."""
    
    __tablename__ = "es_consumer_groups"
    
    # Primary identification
    group_id = Column(String(100), primary_key=True)
    group_name = Column(String(200), nullable=False)
    group_description = Column(Text, nullable=True)
    stream_id = Column(String(100), ForeignKey('es_streams.stream_id'), nullable=True, index=True)
    
    # Group configuration
    session_timeout_ms = Column(Integer, nullable=False, default=30000)
    heartbeat_interval_ms = Column(Integer, nullable=False, default=3000)
    max_poll_interval_ms = Column(Integer, nullable=False, default=300000)
    
    # Rebalancing configuration
    partition_assignment_strategy = Column(String(50), nullable=False, default="round_robin")
    rebalance_timeout_ms = Column(Integer, nullable=False, default=60000)
    
    # Group state
    active_consumers = Column(Integer, nullable=False, default=0)
    total_lag = Column(BigInteger, nullable=False, default=0)
    
    # Multi-tenancy
    tenant_id = Column(String(100), nullable=False, index=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100), nullable=False)
    
    # Relationships
    subscriptions = relationship("ESSubscription", back_populates="consumer_group")
    stream = relationship("ESStream", back_populates="consumer_groups")
    
    # Indexes
    __table_args__ = (
        Index('idx_es_consumer_groups_tenant', 'tenant_id'),
        UniqueConstraint('group_name', 'tenant_id', name='uq_group_name_tenant'),
        CheckConstraint('session_timeout_ms > 0', name='check_session_timeout_positive'),
        CheckConstraint('heartbeat_interval_ms > 0', name='check_heartbeat_positive'),
        CheckConstraint('active_consumers >= 0', name='check_active_consumers_positive')
    )
    
    def __repr__(self):
        return f"<ESConsumerGroup(id={self.group_id}, name={self.group_name}, consumers={self.active_consumers})>"

# =============================================================================
# Schema Registry Model
# =============================================================================

class ESSchema(Base):
    """Event schema registry for validation and evolution."""
    
    __tablename__ = "es_schemas"
    
    # Primary identification
    schema_id = Column(String(100), primary_key=True, default=lambda: f"sch_{uuid7str()}")
    schema_name = Column(String(200), nullable=False)
    schema_version = Column(String(20), nullable=False)
    
    # Schema content
    schema_definition = Column(JSONB, nullable=False)
    schema_format = Column(String(20), nullable=False, default="json_schema")  # json_schema, avro, protobuf
    
    # Event type association
    event_type = Column(String(100), nullable=False, index=True)
    
    # Compatibility settings
    compatibility_level = Column(String(20), nullable=False, default="backward")  # backward, forward, full, none
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Multi-tenancy
    tenant_id = Column(String(100), nullable=False, index=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_es_schemas_tenant', 'tenant_id'),
        Index('idx_es_schemas_event_type', 'event_type'),
        Index('idx_es_schemas_active', 'is_active'),
        UniqueConstraint('schema_name', 'schema_version', 'tenant_id', name='uq_schema_version_tenant')
    )
    
    @validates('schema_definition')
    def validate_schema_definition(self, key, value):
        if not isinstance(value, dict):
            raise ValueError("Schema definition must be a dictionary")
        return value
    
    def __repr__(self):
        return f"<ESSchema(id={self.schema_id}, name={self.schema_name}, version={self.schema_version})>"

# =============================================================================
# Metrics and Monitoring Model
# =============================================================================

class ESMetrics(Base):
    """Event streaming metrics and monitoring data."""
    
    __tablename__ = "es_metrics"
    
    # Primary identification
    metric_id = Column(String(100), primary_key=True, default=lambda: f"met_{uuid7str()}")
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(20), nullable=False)  # counter, gauge, histogram, timer
    
    # Resource association
    stream_id = Column(String(100), ForeignKey("es_streams.stream_id"), nullable=True, index=True)
    consumer_group_id = Column(String(100), nullable=True, index=True)
    
    # Metric data
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)
    
    # Additional metric dimensions
    dimensions = Column(JSONB, nullable=False, default={})
    
    # Time bucket for aggregation
    time_bucket = Column(DateTime(timezone=True), nullable=False, index=True)
    aggregation_period = Column(String(10), nullable=False, default="1m")  # 1m, 5m, 1h, 1d
    
    # Multi-tenancy
    tenant_id = Column(String(100), nullable=False, index=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    stream = relationship("ESStream", back_populates="metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_es_metrics_tenant_time', 'tenant_id', 'time_bucket'),
        Index('idx_es_metrics_name_time', 'metric_name', 'time_bucket'),
        Index('idx_es_metrics_stream_time', 'stream_id', 'time_bucket'),
        Index('idx_es_metrics_group_time', 'consumer_group_id', 'time_bucket'),
        CheckConstraint("metric_type IN ('counter', 'gauge', 'histogram', 'timer')", name='check_metric_type_valid')
    )
    
    def __repr__(self):
        return f"<ESMetrics(id={self.metric_id}, name={self.metric_name}, value={self.metric_value})>"

# =============================================================================
# Audit Log Model
# =============================================================================

class ESAuditLog(Base):
    """Audit logging for event streaming operations."""
    
    __tablename__ = "es_audit_logs"
    
    # Primary identification
    audit_id = Column(String(100), primary_key=True, default=lambda: f"aud_{uuid7str()}")
    
    # Event association
    event_id = Column(String(36), ForeignKey("es_events.event_id"), nullable=True, index=True)
    
    # Operation details
    operation_type = Column(String(50), nullable=False, index=True)  # publish, consume, retry, fail
    operation_status = Column(String(20), nullable=False)  # success, failure, pending
    
    # Actor information
    actor_type = Column(String(20), nullable=False)  # user, system, service
    actor_id = Column(String(100), nullable=False, index=True)
    
    # Operation context
    source_ip = Column(String(45), nullable=True)  # IPv4/IPv6 address
    user_agent = Column(String(500), nullable=True)
    session_id = Column(String(100), nullable=True)
    
    # Operation details
    operation_details = Column(JSONB, nullable=False, default={})
    error_message = Column(Text, nullable=True)
    
    # Multi-tenancy
    tenant_id = Column(String(100), nullable=False, index=True)
    
    # Timing
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    event = relationship("ESEvent", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_es_audit_tenant_time', 'tenant_id', 'created_at'),
        Index('idx_es_audit_operation', 'operation_type', 'created_at'),
        Index('idx_es_audit_actor', 'actor_id', 'created_at'),
        Index('idx_es_audit_status', 'operation_status')
    )
    
    def __repr__(self):
        return f"<ESAuditLog(id={self.audit_id}, operation={self.operation_type}, actor={self.actor_id})>"


# =============================================================================
# ADDITIONAL ENTERPRISE MODELS
# =============================================================================

class ESEventSchema(Base):
	"""Event schema definitions for validation"""
	__tablename__ = "es_event_schemas"
	
	# Primary identification
	schema_id = Column(String(50), primary_key=True, default=uuid7str)
	event_type = Column(String(200), nullable=False, index=True)
	schema_version = Column(String(20), nullable=False, default="1.0")
	
	# Schema definition
	schema_name = Column(String(200), nullable=False)
	schema_description = Column(Text, nullable=True)
	json_schema = Column(JSONB, nullable=False)
	avro_schema = Column(JSONB, nullable=True)
	protobuf_schema = Column(Text, nullable=True)
	
	# Schema metadata
	namespace = Column(String(100), nullable=False, default="default")
	compatibility_level = Column(String(50), nullable=False, default="BACKWARD")
	schema_type = Column(String(50), nullable=False, default="JSON")
	
	# Versioning and evolution
	parent_schema_id = Column(String(50), ForeignKey('es_event_schemas.schema_id'), nullable=True)
	evolution_strategy = Column(String(50), nullable=False, default="BACKWARD_COMPATIBLE")
	is_active = Column(Boolean, nullable=False, default=True)
	is_deprecated = Column(Boolean, nullable=False, default=False)
	deprecation_date = Column(DateTime(timezone=True), nullable=True)
	
	# Validation settings
	strict_validation = Column(Boolean, nullable=False, default=True)
	allow_unknown_fields = Column(Boolean, nullable=False, default=False)
	required_fields = Column(ARRAY(String), nullable=False, default=list)
	optional_fields = Column(ARRAY(String), nullable=False, default=list)
	
	# Usage statistics
	usage_count = Column(BigInteger, nullable=False, default=0)
	last_used = Column(DateTime(timezone=True), nullable=True)
	validation_failures = Column(BigInteger, nullable=False, default=0)
	
	# Multi-tenancy
	tenant_id = Column(String(50), nullable=False, index=True)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	created_by = Column(String(50), nullable=False)
	updated_by = Column(String(50), nullable=False)
	
	# Relationships
	parent_schema = relationship("ESEventSchema", remote_side=[schema_id])
	child_schemas = relationship("ESEventSchema", back_populates="parent_schema")
	
	# Indexes and constraints
	__table_args__ = (
		UniqueConstraint('event_type', 'schema_version', name='uk_es_schemas_type_version'),
		Index('idx_es_schemas_namespace_type', 'namespace', 'event_type'),
		Index('idx_es_schemas_active', 'is_active', 'is_deprecated'),
	)


class ESStreamAssignment(Base):
	"""Assignment of events to streams"""
	__tablename__ = "es_stream_assignments"
	
	# Primary identification
	assignment_id = Column(String(50), primary_key=True, default=uuid7str)
	event_id = Column(String(30), ForeignKey('es_events.event_id'), nullable=False, index=True)
	stream_id = Column(String(100), ForeignKey('es_streams.stream_id'), nullable=False, index=True)
	
	# Kafka details
	partition_id = Column(Integer, nullable=False)
	offset = Column(BigInteger, nullable=False)
	key = Column(String(500), nullable=True)
	
	# Assignment metadata
	assignment_reason = Column(String(100), nullable=False, default="AUTOMATIC")
	assignment_rules = Column(JSONB, nullable=False, default=list)
	
	# Processing tracking
	published_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	consumed_count = Column(Integer, nullable=False, default=0)
	last_consumed_at = Column(DateTime(timezone=True), nullable=True)
	
	# Delivery tracking
	delivery_attempts = Column(Integer, nullable=False, default=0)
	successful_deliveries = Column(Integer, nullable=False, default=0)
	failed_deliveries = Column(Integer, nullable=False, default=0)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	created_by = Column(String(50), nullable=False)
	
	# Relationships
	event = relationship("ESEvent", back_populates="stream_assignments")
	stream = relationship("ESStream", back_populates="assignments")
	
	# Indexes and constraints
	__table_args__ = (
		UniqueConstraint('event_id', 'stream_id', name='uk_es_stream_assignments_event_stream'),
		Index('idx_es_stream_assignments_stream_partition', 'stream_id', 'partition_id'),
		Index('idx_es_stream_assignments_offset', 'stream_id', 'offset'),
		Index('idx_es_stream_assignments_published', 'published_at'),
		CheckConstraint('partition_id >= 0', name='ck_es_stream_assignments_partition_id_non_negative'),
		CheckConstraint('offset >= 0', name='ck_es_stream_assignments_offset_non_negative'),
		CheckConstraint('consumed_count >= 0', name='ck_es_stream_assignments_consumed_count_non_negative'),
	)


class ESEventProcessingHistory(Base):
	"""Event processing history and audit trail"""
	__tablename__ = "es_event_processing_history"
	
	# Primary identification
	history_id = Column(String(50), primary_key=True, default=uuid7str)
	event_id = Column(String(30), ForeignKey('es_events.event_id'), nullable=False, index=True)
	
	# Processing details
	processor_name = Column(String(200), nullable=False)
	processor_version = Column(String(50), nullable=False)
	processing_stage = Column(String(100), nullable=False)
	
	# Status and timing
	status = Column(String(20), nullable=False, index=True)
	started_at = Column(DateTime(timezone=True), nullable=False)
	completed_at = Column(DateTime(timezone=True), nullable=True)
	duration_ms = Column(Integer, nullable=True)
	
	# Processing results
	input_data = Column(JSONB, nullable=True)
	output_data = Column(JSONB, nullable=True)
	transformation_applied = Column(JSONB, nullable=True)
	
	# Error details
	error_message = Column(Text, nullable=True)
	error_code = Column(String(50), nullable=True)
	error_details = Column(JSONB, nullable=True)
	stack_trace = Column(Text, nullable=True)
	
	# Performance metrics
	cpu_time_ms = Column(Integer, nullable=True)
	memory_used_mb = Column(Integer, nullable=True)
	io_operations = Column(Integer, nullable=True)
	
	# Retry information
	retry_attempt = Column(Integer, nullable=False, default=0)
	retry_reason = Column(String(200), nullable=True)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	created_by = Column(String(50), nullable=False)
	
	# Relationships
	event = relationship("ESEvent", back_populates="processing_history")
	
	# Indexes and constraints
	__table_args__ = (
		Index('idx_es_processing_history_event_status', 'event_id', 'status'),
		Index('idx_es_processing_history_processor', 'processor_name', 'processing_stage'),
		Index('idx_es_processing_history_timing', 'started_at', 'completed_at'),
		CheckConstraint('retry_attempt >= 0', name='ck_es_processing_history_retry_attempt_non_negative'),
	)


class ESStreamProcessor(Base):
	"""Stream processing job configuration and management"""
	__tablename__ = "es_stream_processors"
	
	# Primary identification
	processor_id = Column(String(50), primary_key=True, default=uuid7str)
	processor_name = Column(String(200), nullable=False, index=True)
	processor_type = Column(String(50), nullable=False, default=ProcessorType.FILTER)
	
	# Stream association
	stream_id = Column(String(100), ForeignKey('es_streams.stream_id'), nullable=False, index=True)
	output_stream_id = Column(String(100), ForeignKey('es_streams.stream_id'), nullable=True, index=True)
	
	# Processor configuration
	description = Column(Text, nullable=True)
	processing_logic = Column(JSONB, nullable=False)
	configuration = Column(JSONB, nullable=False, default=dict)
	
	# Processing rules and functions
	filter_expression = Column(Text, nullable=True)
	transformation_function = Column(Text, nullable=True)
	aggregation_config = Column(JSONB, nullable=True)
	windowing_config = Column(JSONB, nullable=True)
	
	# Join configuration (for join processors)
	join_stream_id = Column(String(100), ForeignKey('es_streams.stream_id'), nullable=True)
	join_condition = Column(Text, nullable=True)
	join_window_ms = Column(Integer, nullable=True)
	
	# Processing settings
	parallelism = Column(Integer, nullable=False, default=1)
	batch_size = Column(Integer, nullable=False, default=100)
	processing_timeout_ms = Column(Integer, nullable=False, default=30000)
	checkpoint_interval_ms = Column(Integer, nullable=False, default=60000)
	
	# State management
	stateful = Column(Boolean, nullable=False, default=False)
	state_store_config = Column(JSONB, nullable=True)
	changelog_topic = Column(String(300), nullable=True)
	
	# Multi-tenancy and access
	tenant_id = Column(String(50), nullable=False, index=True)
	owner_id = Column(String(50), nullable=False, index=True)
	
	# Status and health
	status = Column(String(20), nullable=False, default="STOPPED", index=True)
	health_status = Column(String(20), nullable=False, default="HEALTHY")
	last_checkpoint = Column(DateTime(timezone=True), nullable=True)
	
	# Processing metrics
	messages_processed = Column(BigInteger, nullable=False, default=0)
	bytes_processed = Column(BigInteger, nullable=False, default=0)
	processing_errors = Column(BigInteger, nullable=False, default=0)
	output_messages = Column(BigInteger, nullable=False, default=0)
	
	# Performance metrics
	throughput_msgs_sec = Column(Integer, nullable=False, default=0)
	latency_p95_ms = Column(Integer, nullable=False, default=0)
	cpu_usage_percent = Column(Integer, nullable=False, default=0)
	memory_usage_mb = Column(Integer, nullable=False, default=0)
	
	# Error handling
	error_tolerance = Column(String(20), nullable=False, default="FAIL")  # FAIL, CONTINUE, SKIP
	dead_letter_enabled = Column(Boolean, nullable=False, default=False)
	dead_letter_topic = Column(String(300), nullable=True)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	created_by = Column(String(50), nullable=False)
	updated_by = Column(String(50), nullable=False)
	
	# Relationships
	stream = relationship("ESStream", back_populates="processors", foreign_keys=[stream_id])
	output_stream = relationship("ESStream", foreign_keys=[output_stream_id])
	join_stream = relationship("ESStream", foreign_keys=[join_stream_id])
	
	# Indexes and constraints
	__table_args__ = (
		Index('idx_es_processors_tenant_type', 'tenant_id', 'processor_type'),
		Index('idx_es_processors_status', 'status'),
		Index('idx_es_processors_stream', 'stream_id', 'status'),
		CheckConstraint('parallelism > 0', name='ck_es_processors_parallelism_positive'),
		CheckConstraint('batch_size > 0', name='ck_es_processors_batch_size_positive'),
		CheckConstraint('messages_processed >= 0', name='ck_es_processors_messages_processed_non_negative'),
	)

# =============================================================================
# Pydantic Models for API
# =============================================================================

class EventConfig(BaseModel):
    """Event configuration model for API."""
    model_config = model_config
    
    event_type: str = Field(..., min_length=1, max_length=100)
    event_version: str = Field(default="1.0", max_length=20)
    source_capability: str = Field(..., min_length=1, max_length=100)
    aggregate_id: str = Field(..., min_length=1, max_length=100)
    aggregate_type: str = Field(..., min_length=1, max_length=100)
    sequence_number: int = Field(default=0, ge=0)
    correlation_id: Optional[str] = Field(None, max_length=100)
    causation_id: Optional[str] = Field(None, max_length=100)
    schema_id: Optional[str] = Field(None, max_length=100)
    schema_version: str = Field(default="1.0", max_length=20)
    partition_key: Optional[str] = Field(None, max_length=200)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StreamConfig(BaseModel):
    """Stream configuration model for API."""
    model_config = model_config
    
    stream_name: str = Field(..., min_length=1, max_length=200)
    stream_description: Optional[str] = Field(None, max_length=1000)
    topic_name: str = Field(..., min_length=1, max_length=200)
    partitions: int = Field(default=3, ge=1, le=1000)
    replication_factor: int = Field(default=3, ge=1, le=10)
    retention_time_ms: int = Field(default=604800000, ge=1)  # 7 days
    retention_size_bytes: Optional[int] = Field(None, ge=1)
    cleanup_policy: str = Field(default="delete")
    compression_type: CompressionType = Field(default=CompressionType.SNAPPY)
    default_serialization: SerializationFormat = Field(default=SerializationFormat.JSON)
    event_category: EventType = Field(default=EventType.DOMAIN_EVENT)
    source_capability: str = Field(..., min_length=1, max_length=100)
    config_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('cleanup_policy')
    @classmethod
    def validate_cleanup_policy(cls, v):
        if v not in ['delete', 'compact']:
            raise ValueError('cleanup_policy must be either "delete" or "compact"')
        return v

class SubscriptionConfig(BaseModel):
    """Subscription configuration model for API."""
    model_config = model_config
    
    subscription_name: str = Field(..., min_length=1, max_length=200)
    subscription_description: Optional[str] = Field(None, max_length=1000)
    stream_id: str = Field(..., min_length=1, max_length=100)
    consumer_group_id: str = Field(..., min_length=1, max_length=100)
    consumer_name: str = Field(..., min_length=1, max_length=200)
    event_type_patterns: List[str] = Field(default_factory=list)
    filter_criteria: Dict[str, Any] = Field(default_factory=dict)
    delivery_mode: DeliveryMode = Field(default=DeliveryMode.AT_LEAST_ONCE)
    batch_size: int = Field(default=100, ge=1, le=10000)
    max_wait_time_ms: int = Field(default=1000, ge=1, le=300000)
    start_position: str = Field(default="latest")
    specific_offset: Optional[int] = Field(None, ge=0)
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {
        "max_retries": 3,
        "retry_delay_ms": 1000,
        "backoff_multiplier": 2.0,
        "max_delay_ms": 60000
    })
    dead_letter_enabled: bool = Field(default=True)
    dead_letter_topic: Optional[str] = Field(None, max_length=200)
    webhook_url: Optional[str] = Field(None, max_length=500)
    webhook_headers: Dict[str, str] = Field(default_factory=dict)
    webhook_timeout_ms: Optional[int] = Field(None, ge=1000, le=300000)
    
    @field_validator('start_position')
    @classmethod
    def validate_start_position(cls, v):
        if v not in ['earliest', 'latest', 'specific_offset']:
            raise ValueError('start_position must be "earliest", "latest", or "specific_offset"')
        return v

class SchemaConfig(BaseModel):
    """Schema configuration model for API."""
    model_config = model_config
    
    schema_name: str = Field(..., min_length=1, max_length=200)
    schema_version: str = Field(..., min_length=1, max_length=20)
    schema_definition: Dict[str, Any] = Field(...)
    schema_format: str = Field(default="json_schema")
    event_type: str = Field(..., min_length=1, max_length=100)
    compatibility_level: str = Field(default="backward")
    
    @field_validator('schema_format')
    @classmethod
    def validate_schema_format(cls, v):
        if v not in ['json_schema', 'avro', 'protobuf']:
            raise ValueError('schema_format must be "json_schema", "avro", or "protobuf"')
        return v
    
    @field_validator('compatibility_level')
    @classmethod
    def validate_compatibility_level(cls, v):
        if v not in ['backward', 'forward', 'full', 'none']:
            raise ValueError('compatibility_level must be "backward", "forward", "full", or "none"')
        return v

# =============================================================================
# ENHANCED PYDANTIC API MODELS
# =============================================================================

class EventCreate(BaseModel):
	"""Pydantic model for creating events"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	event_type: str = Field(..., max_length=200, description="Event type identifier")
	event_version: str = Field(default="1.0", max_length=20, description="Event schema version")
	source_capability: str = Field(..., max_length=100, description="Source capability identifier")
	target_capability: Optional[str] = Field(None, max_length=100, description="Target capability identifier")
	aggregate_id: str = Field(..., max_length=50, description="Aggregate identifier")
	aggregate_type: str = Field(..., max_length=100, description="Aggregate type")
	sequence_number: int = Field(..., gt=0, description="Event sequence number")
	correlation_id: Optional[str] = Field(None, max_length=50, description="Correlation ID for related events")
	causation_id: Optional[str] = Field(None, max_length=30, description="ID of the event that caused this event")
	tenant_id: str = Field(..., max_length=50, description="Tenant identifier")
	user_id: Optional[str] = Field(None, max_length=50, description="User identifier")
	session_id: Optional[str] = Field(None, max_length=50, description="Session identifier")
	priority: EventPriority = Field(default=EventPriority.NORMAL, description="Event priority level")
	payload: Dict[str, Any] = Field(..., description="Event payload data")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")
	headers: Dict[str, Any] = Field(default_factory=dict, description="Event headers")
	schema_id: Optional[str] = Field(None, max_length=50, description="Schema ID for validation")
	content_type: str = Field(default="application/json", max_length=100, description="Content type")
	created_by: str = Field(..., max_length=50, description="Creator identifier")

class EventResponse(BaseModel):
	"""Pydantic model for event responses"""
	model_config = ConfigDict(from_attributes=True)
	
	event_id: str
	event_type: str
	event_version: str
	source_capability: str
	target_capability: Optional[str]
	aggregate_id: str
	aggregate_type: str
	sequence_number: int
	correlation_id: Optional[str]
	causation_id: Optional[str]
	event_timestamp: datetime
	tenant_id: str
	user_id: Optional[str]
	status: EventStatus
	priority: EventPriority
	payload: Dict[str, Any]
	metadata: Dict[str, Any]
	headers: Dict[str, Any]
	schema_version: str
	created_at: datetime
	updated_at: datetime

class StreamCreate(BaseModel):
	"""Pydantic model for creating streams"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	stream_name: str = Field(..., max_length=200, description="Stream name")
	stream_type: str = Field(default="EVENT_STREAM", max_length=50, description="Stream type")
	description: Optional[str] = Field(None, description="Stream description")
	stream_category: str = Field(default="general", max_length=100, description="Stream category")
	business_domain: str = Field(default="default", max_length=100, description="Business domain")
	topic_name: str = Field(..., max_length=300, description="Kafka topic name")
	partition_count: int = Field(default=12, gt=0, description="Number of partitions")
	replication_factor: int = Field(default=3, gt=0, description="Replication factor")
	min_in_sync_replicas: int = Field(default=2, gt=0, description="Minimum in-sync replicas")
	retention_time_ms: int = Field(default=604800000, gt=0, description="Retention time in milliseconds")
	retention_size_bytes: Optional[int] = Field(None, gt=0, description="Retention size in bytes")
	cleanup_policy: str = Field(default="delete", description="Cleanup policy")
	compression_type: CompressionType = Field(default=CompressionType.SNAPPY, description="Compression type")
	serialization_format: SerializationFormat = Field(default=SerializationFormat.JSON, description="Serialization format")
	tenant_id: str = Field(..., max_length=50, description="Tenant identifier")
	owner_id: str = Field(..., max_length=50, description="Owner identifier")
	visibility: str = Field(default="PRIVATE", description="Stream visibility")
	encryption_enabled: bool = Field(default=True, description="Enable encryption")
	access_control_enabled: bool = Field(default=True, description="Enable access control")
	event_filters: List[Dict[str, Any]] = Field(default_factory=list, description="Event filters")
	routing_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Routing rules")
	created_by: str = Field(..., max_length=50, description="Creator identifier")

class StreamResponse(BaseModel):
	"""Pydantic model for stream responses"""
	model_config = ConfigDict(from_attributes=True)
	
	stream_id: str
	stream_name: str
	stream_type: str
	description: Optional[str]
	stream_category: str
	business_domain: str
	topic_name: str
	partition_count: int
	replication_factor: int
	status: StreamStatus
	health_status: str
	tenant_id: str
	owner_id: str
	visibility: str
	encryption_enabled: bool
	message_count: int
	bytes_in: int
	bytes_out: int
	producer_count: int
	consumer_count: int
	created_at: datetime
	updated_at: datetime

class ConsumerGroupCreate(BaseModel):
	"""Pydantic model for creating consumer groups"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	group_name: str = Field(..., max_length=200, description="Consumer group name")
	stream_id: str = Field(..., max_length=50, description="Stream identifier")
	description: Optional[str] = Field(None, description="Consumer group description")
	group_type: str = Field(default="STANDARD", description="Consumer group type")
	processing_mode: str = Field(default="PARALLEL", description="Processing mode")
	max_consumers: int = Field(default=10, gt=0, description="Maximum number of consumers")
	auto_offset_reset: str = Field(default="latest", description="Auto offset reset policy")
	enable_auto_commit: bool = Field(default=True, description="Enable auto commit")
	delivery_guarantee: DeliveryMode = Field(default=DeliveryMode.AT_LEAST_ONCE, description="Delivery guarantee")
	enable_idempotency: bool = Field(default=False, description="Enable idempotency")
	retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_retries": 3, "backoff_ms": 1000}, description="Retry policy")
	dead_letter_enabled: bool = Field(default=True, description="Enable dead letter queue")
	event_filters: List[Dict[str, Any]] = Field(default_factory=list, description="Event filters")
	processing_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Processing rules")
	tenant_id: str = Field(..., max_length=50, description="Tenant identifier")
	owner_id: str = Field(..., max_length=50, description="Owner identifier")
	created_by: str = Field(..., max_length=50, description="Creator identifier")

class ConsumerGroupResponse(BaseModel):
	"""Pydantic model for consumer group responses"""
	model_config = ConfigDict(from_attributes=True)
	
	group_id: str
	group_name: str
	stream_id: str
	description: Optional[str]
	group_type: str
	processing_mode: str
	max_consumers: int
	delivery_guarantee: DeliveryMode
	status: ConsumerStatus
	health_status: str
	active_consumers: int
	messages_consumed: int
	bytes_consumed: int
	processing_errors: int
	consumer_lag: int
	tenant_id: str
	owner_id: str
	created_at: datetime
	updated_at: datetime

class StreamProcessorCreate(BaseModel):
	"""Pydantic model for creating stream processors"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	processor_name: str = Field(..., max_length=200, description="Processor name")
	processor_type: ProcessorType = Field(..., description="Processor type")
	stream_id: str = Field(..., max_length=50, description="Input stream identifier")
	output_stream_id: Optional[str] = Field(None, max_length=50, description="Output stream identifier")
	description: Optional[str] = Field(None, description="Processor description")
	processing_logic: Dict[str, Any] = Field(..., description="Processing logic configuration")
	configuration: Dict[str, Any] = Field(default_factory=dict, description="Processor configuration")
	filter_expression: Optional[str] = Field(None, description="Filter expression")
	transformation_function: Optional[str] = Field(None, description="Transformation function")
	parallelism: int = Field(default=1, gt=0, description="Processing parallelism")
	batch_size: int = Field(default=100, gt=0, description="Batch size")
	processing_timeout_ms: int = Field(default=30000, gt=0, description="Processing timeout")
	stateful: bool = Field(default=False, description="Is processor stateful")
	tenant_id: str = Field(..., max_length=50, description="Tenant identifier")
	owner_id: str = Field(..., max_length=50, description="Owner identifier")
	created_by: str = Field(..., max_length=50, description="Creator identifier")

class StreamProcessorResponse(BaseModel):
	"""Pydantic model for stream processor responses"""
	model_config = ConfigDict(from_attributes=True)
	
	processor_id: str
	processor_name: str
	processor_type: ProcessorType
	stream_id: str
	output_stream_id: Optional[str]
	description: Optional[str]
	parallelism: int
	batch_size: int
	stateful: bool
	status: str
	health_status: str
	messages_processed: int
	bytes_processed: int
	processing_errors: int
	output_messages: int
	throughput_msgs_sec: int
	latency_p95_ms: int
	tenant_id: str
	owner_id: str
	created_at: datetime
	updated_at: datetime

# Export all models for external use
__all__ = [
    # SQLAlchemy models
    "Base",
    "ESEvent",
    "ESStream", 
    "ESSubscription",
    "ESConsumerGroup",
    "ESSchema",
    "ESMetrics",
    "ESAuditLog",
    "ESEventSchema",
    "ESStreamAssignment",
    "ESEventProcessingHistory",
    "ESStreamProcessor",
    
    # Enums
    "EventStatus",
    "EventPriority",
    "StreamStatus",
    "SubscriptionStatus",
    "ConsumerStatus",
    "ProcessorType",
    "EventType",
    "DeliveryMode",
    "CompressionType",
    "SerializationFormat",
    
    # Legacy Pydantic models
    "EventConfig",
    "StreamConfig",
    "SubscriptionConfig",
    "SchemaConfig",
    
    # Enhanced Pydantic models
    "EventCreate",
    "EventResponse",
    "StreamCreate",
    "StreamResponse",
    "ConsumerGroupCreate",
    "ConsumerGroupResponse",
    "StreamProcessorCreate",
    "StreamProcessorResponse"
]
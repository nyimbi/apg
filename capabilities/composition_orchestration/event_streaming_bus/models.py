"""
APG Event Streaming Bus - Data Models

Comprehensive data models for event streaming, including events, streams,
subscriptions, schemas, and metrics with full SQLAlchemy and Pydantic integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import JSONB, UUID

from pydantic import (
    BaseModel, Field, ConfigDict, validator, root_validator,
    AfterValidator, field_validator
)
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
    """Event processing status."""
    PENDING = "pending"
    PUBLISHED = "published"
    CONSUMED = "consumed"
    FAILED = "failed"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"

class StreamStatus(str, Enum):
    """Stream operational status."""
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    ERROR = "error"

class SubscriptionStatus(str, Enum):
    """Subscription status."""
    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    ERROR = "error"

class EventType(str, Enum):
    """Event type categories."""
    DOMAIN_EVENT = "domain_event"
    INTEGRATION_EVENT = "integration_event"
    NOTIFICATION_EVENT = "notification_event"
    SYSTEM_EVENT = "system_event"
    AUDIT_EVENT = "audit_event"

class DeliveryMode(str, Enum):
    """Message delivery guarantees."""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"

class CompressionType(str, Enum):
    """Stream compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"

class SerializationFormat(str, Enum):
    """Event serialization formats."""
    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    MESSAGEPACK = "messagepack"

# =============================================================================
# Core Event Model
# =============================================================================

class ESEvent(Base):
    """Core event model for the event streaming platform."""
    
    __tablename__ = "es_events"
    
    # Primary identification
    event_id = Column(String(36), primary_key=True, default=lambda: f"evt_{uuid7str()}")
    event_type = Column(String(100), nullable=False, index=True)
    event_version = Column(String(20), nullable=False, default="1.0")
    
    # Event source and aggregate information
    source_capability = Column(String(100), nullable=False, index=True)
    aggregate_id = Column(String(100), nullable=False, index=True)
    aggregate_type = Column(String(100), nullable=False)
    sequence_number = Column(BigInteger, nullable=False, default=0)
    
    # Timing and correlation
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    correlation_id = Column(String(100), nullable=True, index=True)
    causation_id = Column(String(100), nullable=True, index=True)
    
    # Multi-tenancy and security
    tenant_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    
    # Event content
    payload = Column(JSONB, nullable=False)
    metadata = Column(JSONB, nullable=True, default={})
    
    # Schema and format
    schema_id = Column(String(100), nullable=True)
    schema_version = Column(String(20), nullable=False, default="1.0")
    serialization_format = Column(String(20), nullable=False, default=SerializationFormat.JSON.value)
    
    # Processing status
    status = Column(String(20), nullable=False, default=EventStatus.PENDING.value)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    
    # Stream assignment
    stream_id = Column(String(100), nullable=False, index=True)
    partition_key = Column(String(200), nullable=True)
    offset_position = Column(BigInteger, nullable=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100), nullable=False)
    
    # Relationships
    stream = relationship("ESStream", back_populates="events")
    audit_logs = relationship("ESAuditLog", back_populates="event")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_es_events_tenant_type', 'tenant_id', 'event_type'),
        Index('idx_es_events_aggregate', 'aggregate_type', 'aggregate_id'),
        Index('idx_es_events_timestamp', 'timestamp'),
        Index('idx_es_events_correlation', 'correlation_id'),
        Index('idx_es_events_stream_offset', 'stream_id', 'offset_position'),
        Index('idx_es_events_sequence', 'aggregate_id', 'sequence_number'),
        CheckConstraint('retry_count >= 0', name='check_retry_count_positive'),
        CheckConstraint('max_retries >= 0', name='check_max_retries_positive'),
        CheckConstraint('sequence_number >= 0', name='check_sequence_positive')
    )
    
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
    
    # Enums
    "EventStatus",
    "StreamStatus",
    "SubscriptionStatus", 
    "EventType",
    "DeliveryMode",
    "CompressionType",
    "SerializationFormat",
    
    # Pydantic models
    "EventConfig",
    "StreamConfig",
    "SubscriptionConfig",
    "SchemaConfig"
]
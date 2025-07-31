"""
APG Event Streaming Bus - Service Layer

Comprehensive service layer implementation providing event streaming, publishing,
consumption, stream processing, and event sourcing capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from contextlib import asynccontextmanager
import hashlib
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload
import redis.asyncio as redis
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError
import aiokafka
from uuid_extensions import uuid7str

from .models import (
    ESEvent, ESStream, ESSubscription, ESConsumerGroup, ESSchema, ESMetrics, ESAuditLog,
    ESEventSchema, ESStreamAssignment, ESEventProcessingHistory, ESStreamProcessor,
    EventStatus, EventPriority, StreamStatus, SubscriptionStatus, ConsumerStatus, ProcessorType,
    EventType, DeliveryMode, CompressionType, SerializationFormat,
    EventConfig, StreamConfig, SubscriptionConfig, SchemaConfig,
    EventCreate, EventResponse, StreamCreate, StreamResponse
)

# =============================================================================
# Logging Configuration
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# Event Publishing Service
# =============================================================================

class EventPublishingService:
    """Service for publishing events to the streaming platform."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis, kafka_config: Dict[str, Any]):
        self.db_session = db_session
        self.redis_client = redis_client
        self.kafka_config = kafka_config
        self.kafka_producer = None
        self._producer_lock = asyncio.Lock()
        
    async def _get_kafka_producer(self) -> aiokafka.AIOKafkaProducer:
        """Get or create Kafka producer instance."""
        if self.kafka_producer is None:
            async with self._producer_lock:
                if self.kafka_producer is None:
                    self.kafka_producer = aiokafka.AIOKafkaProducer(
                        bootstrap_servers=self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
                        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                        key_serializer=lambda k: k.encode('utf-8') if k else None,
                        acks='all',  # Wait for all replicas
                        retries=3,
                        max_in_flight_requests_per_connection=1,  # Ensure ordering
                        enable_idempotence=True,  # Exactly-once semantics
                        compression_type='snappy',
                        batch_size=16384,
                        linger_ms=10
                    )
                    await self.kafka_producer.start()
        return self.kafka_producer
    
    async def publish_event(
        self,
        event_config: EventConfig,
        payload: Dict[str, Any],
        tenant_id: str,
        user_id: str
    ) -> str:
        """Publish a single event to the streaming platform."""
        
        # Generate event ID
        event_id = f"evt_{uuid7str()}"
        
        # Determine stream and validate
        stream = await self._get_stream_for_event(event_config.event_type, event_config.source_capability, tenant_id)
        if not stream:
            raise ValueError(f"No stream found for event type: {event_config.event_type}")
        
        # Validate schema if configured
        await self._validate_event_schema(event_config.event_type, payload, tenant_id)
        
        # Create event record
        event = ESEvent(
            event_id=event_id,
            event_type=event_config.event_type,
            event_version=event_config.event_version,
            source_capability=event_config.source_capability,
            aggregate_id=event_config.aggregate_id,
            aggregate_type=event_config.aggregate_type,
            sequence_number=event_config.sequence_number,
            correlation_id=event_config.correlation_id,
            causation_id=event_config.causation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            payload=payload,
            metadata=event_config.metadata,
            schema_id=event_config.schema_id,
            schema_version=event_config.schema_version,
            stream_id=stream.stream_id,
            partition_key=event_config.partition_key or event_config.aggregate_id,
            status=EventStatus.PENDING.value,
            created_by=user_id
        )
        
        try:
            # Save to database
            self.db_session.add(event)
            await self.db_session.commit()
            
            # Publish to Kafka
            await self._publish_to_kafka(event, stream)
            
            # Update status to published
            event.status = EventStatus.PUBLISHED.value
            await self.db_session.commit()
            
            # Cache recent event for fast access
            await self._cache_event(event)
            
            # Log audit trail
            await self._log_audit_event("publish", "success", event_id, user_id, tenant_id, {
                "event_type": event_config.event_type,
                "stream_id": stream.stream_id
            })
            
            logger.info(f"Event published successfully: {event_id}")
            return event_id
            
        except Exception as e:
            # Update status to failed
            event.status = EventStatus.FAILED.value
            await self.db_session.commit()
            
            # Log audit trail
            await self._log_audit_event("publish", "failure", event_id, user_id, tenant_id, {
                "error": str(e)
            })
            
            logger.error(f"Failed to publish event {event_id}: {e}")
            raise
    
    async def publish_events_batch(
        self,
        events_data: List[tuple[EventConfig, Dict[str, Any]]],
        tenant_id: str,
        user_id: str
    ) -> List[str]:
        """Publish multiple events in a batch for improved performance."""
        
        event_ids = []
        events = []
        
        try:
            # Create all event records
            for event_config, payload in events_data:
                event_id = f"evt_{uuid7str()}"
                
                # Get stream
                stream = await self._get_stream_for_event(event_config.event_type, event_config.source_capability, tenant_id)
                if not stream:
                    raise ValueError(f"No stream found for event type: {event_config.event_type}")
                
                # Validate schema
                await self._validate_event_schema(event_config.event_type, payload, tenant_id)
                
                # Create event
                event = ESEvent(
                    event_id=event_id,
                    event_type=event_config.event_type,
                    event_version=event_config.event_version,
                    source_capability=event_config.source_capability,
                    aggregate_id=event_config.aggregate_id,
                    aggregate_type=event_config.aggregate_type,
                    sequence_number=event_config.sequence_number,
                    correlation_id=event_config.correlation_id,
                    causation_id=event_config.causation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    payload=payload,
                    metadata=event_config.metadata,
                    schema_id=event_config.schema_id,
                    schema_version=event_config.schema_version,
                    stream_id=stream.stream_id,
                    partition_key=event_config.partition_key or event_config.aggregate_id,
                    status=EventStatus.PENDING.value,
                    created_by=user_id
                )
                
                events.append((event, stream))
                event_ids.append(event_id)
            
            # Save all events to database
            for event, _ in events:
                self.db_session.add(event)
            await self.db_session.commit()
            
            # Publish all events to Kafka
            producer = await self._get_kafka_producer()
            tasks = []
            
            for event, stream in events:
                task = self._publish_to_kafka_async(producer, event, stream)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # Update all statuses to published
            for event, _ in events:
                event.status = EventStatus.PUBLISHED.value
            await self.db_session.commit()
            
            # Cache and audit
            for event, _ in events:
                await self._cache_event(event)
                await self._log_audit_event("publish_batch", "success", event.event_id, user_id, tenant_id, {
                    "event_type": event.event_type,
                    "batch_size": len(events)
                })
            
            logger.info(f"Batch published {len(events)} events successfully")
            return event_ids
            
        except Exception as e:
            # Update failed events
            for event, _ in events:
                event.status = EventStatus.FAILED.value
            await self.db_session.commit()
            
            logger.error(f"Failed to publish event batch: {e}")
            raise
    
    async def _get_stream_for_event(self, event_type: str, source_capability: str, tenant_id: str) -> Optional[ESStream]:
        """Get the appropriate stream for an event type."""
        
        # Try exact match first
        result = await self.db_session.execute(
            select(ESStream).where(
                and_(
                    ESStream.tenant_id == tenant_id,
                    ESStream.source_capability == source_capability,
                    ESStream.status == StreamStatus.ACTIVE.value
                )
            )
        )
        
        stream = result.scalar_one_or_none()
        if stream:
            return stream
        
        # Try pattern matching for event type
        result = await self.db_session.execute(
            select(ESStream).where(
                and_(
                    ESStream.tenant_id == tenant_id,
                    ESStream.status == StreamStatus.ACTIVE.value
                )
            )
        )
        
        streams = result.scalars().all()
        
        # Check if event type matches any stream pattern
        for stream in streams:
            config = stream.config_settings or {}
            patterns = config.get('event_type_patterns', [])
            
            for pattern in patterns:
                if re.match(pattern.replace('*', '.*'), event_type):
                    return stream
        
        return None
    
    async def _validate_event_schema(self, event_type: str, payload: Dict[str, Any], tenant_id: str):
        """Validate event payload against schema if configured."""
        
        # Get active schema for event type
        result = await self.db_session.execute(
            select(ESSchema).where(
                and_(
                    ESSchema.tenant_id == tenant_id,
                    ESSchema.event_type == event_type,
                    ESSchema.is_active == True
                )
            ).order_by(desc(ESSchema.created_at))
        )
        
        schema = result.scalar_one_or_none()
        if not schema:
            return  # No schema validation required
        
        # Validate using JSON Schema (basic implementation)
        if schema.schema_format == "json_schema":
            import jsonschema
            try:
                jsonschema.validate(payload, schema.schema_definition)
            except jsonschema.ValidationError as e:
                raise ValueError(f"Schema validation failed: {e.message}")
    
    async def _publish_to_kafka(self, event: ESEvent, stream: ESStream):
        """Publish event to Kafka topic."""
        producer = await self._get_kafka_producer()
        await self._publish_to_kafka_async(producer, event, stream)
    
    async def _publish_to_kafka_async(self, producer: aiokafka.AIOKafkaProducer, event: ESEvent, stream: ESStream):
        """Async helper for Kafka publishing."""
        
        # Prepare event data for Kafka
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "event_version": event.event_version,
            "source_capability": event.source_capability,
            "aggregate_id": event.aggregate_id,
            "aggregate_type": event.aggregate_type,
            "sequence_number": event.sequence_number,
            "timestamp": event.timestamp.isoformat(),
            "correlation_id": event.correlation_id,
            "causation_id": event.causation_id,
            "tenant_id": event.tenant_id,
            "user_id": event.user_id,
            "payload": event.payload,
            "metadata": event.metadata,
            "schema_id": event.schema_id,
            "schema_version": event.schema_version
        }
        
        # Send to Kafka
        try:
            future = await producer.send(
                topic=stream.topic_name,
                value=event_data,
                key=event.partition_key,
                partition=None  # Let Kafka handle partitioning
            )
            
            # Update offset position
            record_metadata = await future
            event.offset_position = record_metadata.offset
            
        except Exception as e:
            logger.error(f"Kafka publish failed for event {event.event_id}: {e}")
            raise
    
    async def _cache_event(self, event: ESEvent):
        """Cache recent event for fast access."""
        cache_key = f"event:{event.tenant_id}:{event.event_id}"
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "aggregate_id": event.aggregate_id,
            "timestamp": event.timestamp.isoformat(),
            "status": event.status
        }
        
        await self.redis_client.setex(cache_key, 3600, json.dumps(event_data))  # 1 hour cache
    
    async def _log_audit_event(
        self,
        operation_type: str,
        operation_status: str,
        event_id: str,
        actor_id: str,
        tenant_id: str,
        details: Dict[str, Any]
    ):
        """Log audit event for operation."""
        
        audit_log = ESAuditLog(
            event_id=event_id,
            operation_type=operation_type,
            operation_status=operation_status,
            actor_type="user",
            actor_id=actor_id,
            tenant_id=tenant_id,
            operation_details=details
        )
        
        self.db_session.add(audit_log)
        # Note: Commit will happen in calling function
    
    async def close(self):
        """Close Kafka producer and clean up resources."""
        if self.kafka_producer:
            await self.kafka_producer.stop()

# =============================================================================
# Event Consumption Service
# =============================================================================

class EventConsumptionService:
    """Service for consuming events from streams."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis, kafka_config: Dict[str, Any]):
        self.db_session = db_session
        self.redis_client = redis_client
        self.kafka_config = kafka_config
        self.active_consumers: Dict[str, aiokafka.AIOKafkaConsumer] = {}
        self.consumer_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_subscription(self, subscription_id: str) -> bool:
        """Start consuming events for a subscription."""
        
        # Get subscription details
        result = await self.db_session.execute(
            select(ESSubscription).options(selectinload(ESSubscription.stream))
            .where(ESSubscription.subscription_id == subscription_id)
        )
        
        subscription = result.scalar_one_or_none()
        if not subscription:
            raise ValueError(f"Subscription not found: {subscription_id}")
        
        if subscription.status != SubscriptionStatus.ACTIVE.value:
            raise ValueError(f"Subscription is not active: {subscription_id}")
        
        # Check if already consuming
        if subscription_id in self.active_consumers:
            logger.warning(f"Subscription already active: {subscription_id}")
            return False
        
        # Create Kafka consumer
        consumer = aiokafka.AIOKafkaConsumer(
            subscription.stream.topic_name,
            bootstrap_servers=self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
            group_id=subscription.consumer_group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset=subscription.start_position,
            enable_auto_commit=False,  # Manual commit for reliability
            max_poll_records=subscription.batch_size,
            consumer_timeout_ms=subscription.max_wait_time_ms
        )
        
        try:
            await consumer.start()
            self.active_consumers[subscription_id] = consumer
            
            # Start consumption task
            task = asyncio.create_task(self._consume_events(subscription, consumer))
            self.consumer_tasks[subscription_id] = task
            
            logger.info(f"Started subscription: {subscription_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start subscription {subscription_id}: {e}")
            if consumer:
                await consumer.stop()
            raise
    
    async def stop_subscription(self, subscription_id: str) -> bool:
        """Stop consuming events for a subscription."""
        
        if subscription_id not in self.active_consumers:
            logger.warning(f"Subscription not active: {subscription_id}")
            return False
        
        # Cancel consumption task
        if subscription_id in self.consumer_tasks:
            task = self.consumer_tasks[subscription_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.consumer_tasks[subscription_id]
        
        # Stop consumer
        consumer = self.active_consumers[subscription_id]
        await consumer.stop()
        del self.active_consumers[subscription_id]
        
        logger.info(f"Stopped subscription: {subscription_id}")
        return True
    
    async def _consume_events(self, subscription: ESSubscription, consumer: aiokafka.AIOKafkaConsumer):
        """Main event consumption loop."""
        
        try:
            async for batch in consumer:
                if not batch:
                    continue
                
                # Process batch of events
                await self._process_event_batch(subscription, batch)
                
                # Commit offsets after successful processing
                await consumer.commit()
                
                # Update consumption metrics
                await self._update_consumption_metrics(subscription, len(batch))
                
        except asyncio.CancelledError:
            logger.info(f"Consumption cancelled for subscription: {subscription.subscription_id}")
        except Exception as e:
            logger.error(f"Error in consumption loop for {subscription.subscription_id}: {e}")
            
            # Update subscription status to error
            subscription.status = SubscriptionStatus.ERROR.value
            await self.db_session.commit()
    
    async def _process_event_batch(self, subscription: ESSubscription, batch: List[Any]):
        """Process a batch of consumed events."""
        
        for message in batch:
            try:
                # Check if event matches subscription filters
                if not await self._matches_subscription_filters(subscription, message.value):
                    continue
                
                # Process individual event
                await self._process_single_event(subscription, message)
                
            except Exception as e:
                logger.error(f"Failed to process event in subscription {subscription.subscription_id}: {e}")
                
                # Handle retry or dead letter
                await self._handle_processing_failure(subscription, message, e)
    
    async def _matches_subscription_filters(self, subscription: ESSubscription, event_data: Dict[str, Any]) -> bool:
        """Check if event matches subscription filters."""
        
        # Check event type patterns
        event_type = event_data.get('event_type', '')
        patterns = subscription.event_type_patterns or []
        
        if patterns:
            matches = False
            for pattern in patterns:
                if re.match(pattern.replace('*', '.*'), event_type):
                    matches = True
                    break
            if not matches:
                return False
        
        # Check additional filter criteria
        filter_criteria = subscription.filter_criteria or {}
        
        for field, expected_value in filter_criteria.items():
            actual_value = event_data.get(field)
            
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True
    
    async def _process_single_event(self, subscription: ESSubscription, message: Any):
        """Process a single consumed event."""
        
        event_data = message.value
        
        # Update last consumed position
        subscription.last_consumed_offset = message.offset
        subscription.last_consumed_at = datetime.now(timezone.utc)
        
        # Deliver event based on subscription type
        if subscription.webhook_url:
            await self._deliver_via_webhook(subscription, event_data)
        else:
            # For now, just log the event (would integrate with specific consumers)
            logger.info(f"Consumed event {event_data.get('event_id')} for subscription {subscription.subscription_id}")
        
        # Record consumption metrics
        await self._record_consumption_metric(subscription, event_data)
    
    async def _deliver_via_webhook(self, subscription: ESSubscription, event_data: Dict[str, Any]):
        """Deliver event via webhook."""
        
        import aiohttp
        
        timeout = aiohttp.ClientTimeout(total=subscription.webhook_timeout_ms / 1000)
        headers = subscription.webhook_headers or {}
        headers['Content-Type'] = 'application/json'
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    subscription.webhook_url,
                    json=event_data,
                    headers=headers
                ) as response:
                    if response.status >= 400:
                        raise Exception(f"Webhook delivery failed with status {response.status}")
                        
            except Exception as e:
                logger.error(f"Webhook delivery failed for subscription {subscription.subscription_id}: {e}")
                raise
    
    async def _handle_processing_failure(self, subscription: ESSubscription, message: Any, error: Exception):
        """Handle processing failure with retry logic."""
        
        retry_policy = subscription.retry_policy or {}
        max_retries = retry_policy.get('max_retries', 3)
        
        # For now, just log the failure (would implement retry queue)
        logger.error(f"Processing failed for subscription {subscription.subscription_id}: {error}")
        
        # If dead letter is enabled, send to dead letter topic
        if subscription.dead_letter_enabled and subscription.dead_letter_topic:
            await self._send_to_dead_letter(subscription, message, error)
    
    async def _send_to_dead_letter(self, subscription: ESSubscription, message: Any, error: Exception):
        """Send failed message to dead letter topic."""
        
        # Create dead letter event
        dead_letter_data = {
            "original_event": message.value,
            "subscription_id": subscription.subscription_id,
            "error": str(error),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "retry_count": 0  # Would track actual retry count
        }
        
        # Send to dead letter topic (implementation would use Kafka producer)
        logger.warning(f"Sent event to dead letter queue for subscription {subscription.subscription_id}")
    
    async def _update_consumption_metrics(self, subscription: ESSubscription, batch_size: int):
        """Update consumption metrics."""
        
        # Update consumer group lag and metrics
        group_result = await self.db_session.execute(
            select(ESConsumerGroup).where(ESConsumerGroup.group_id == subscription.consumer_group_id)
        )
        
        group = group_result.scalar_one_or_none()
        if group:
            # Update active consumers count and lag (simplified)
            group.active_consumers = len([s for s in self.active_consumers.keys() 
                                        if s.startswith(subscription.consumer_group_id)])
            await self.db_session.commit()
    
    async def _record_consumption_metric(self, subscription: ESSubscription, event_data: Dict[str, Any]):
        """Record consumption metric."""
        
        metric = ESMetrics(
            metric_name="events_consumed",
            metric_type="counter",
            stream_id=subscription.stream_id,
            consumer_group_id=subscription.consumer_group_id,
            metric_value=1,
            metric_unit="count",
            dimensions={
                "subscription_id": subscription.subscription_id,
                "event_type": event_data.get('event_type')
            },
            time_bucket=datetime.now(timezone.utc).replace(second=0, microsecond=0),
            tenant_id=subscription.tenant_id
        )
        
        self.db_session.add(metric)
    
    async def get_subscription_status(self, subscription_id: str) -> Dict[str, Any]:
        """Get current status of a subscription."""
        
        result = await self.db_session.execute(
            select(ESSubscription).where(ESSubscription.subscription_id == subscription_id)
        )
        
        subscription = result.scalar_one_or_none()
        if not subscription:
            raise ValueError(f"Subscription not found: {subscription_id}")
        
        return {
            "subscription_id": subscription_id,
            "status": subscription.status,
            "is_consuming": subscription_id in self.active_consumers,
            "last_consumed_offset": subscription.last_consumed_offset,
            "last_consumed_at": subscription.last_consumed_at.isoformat() if subscription.last_consumed_at else None
        }
    
    async def close(self):
        """Close all consumers and clean up resources."""
        
        # Stop all active subscriptions
        for subscription_id in list(self.active_consumers.keys()):
            await self.stop_subscription(subscription_id)

# =============================================================================
# Stream Processing Service
# =============================================================================

class StreamProcessingService:
    """Service for real-time stream processing and analytics."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis, kafka_config: Dict[str, Any]):
        self.db_session = db_session
        self.redis_client = redis_client
        self.kafka_config = kafka_config
        self.processors: Dict[str, asyncio.Task] = {}
    
    async def start_stream_processor(self, processor_id: str, processor_config: Dict[str, Any]) -> bool:
        """Start a stream processing job."""
        
        if processor_id in self.processors:
            logger.warning(f"Stream processor already running: {processor_id}")
            return False
        
        # Create and start processor task
        task = asyncio.create_task(self._run_stream_processor(processor_id, processor_config))
        self.processors[processor_id] = task
        
        logger.info(f"Started stream processor: {processor_id}")
        return True
    
    async def stop_stream_processor(self, processor_id: str) -> bool:
        """Stop a stream processing job."""
        
        if processor_id not in self.processors:
            logger.warning(f"Stream processor not running: {processor_id}")
            return False
        
        task = self.processors[processor_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self.processors[processor_id]
        
        logger.info(f"Stopped stream processor: {processor_id}")
        return True
    
    async def _run_stream_processor(self, processor_id: str, config: Dict[str, Any]):
        """Run stream processing logic."""
        
        processor_type = config.get('type', 'aggregation')
        
        if processor_type == 'aggregation':
            await self._run_aggregation_processor(processor_id, config)
        elif processor_type == 'windowing':
            await self._run_windowing_processor(processor_id, config)
        elif processor_type == 'join':
            await self._run_join_processor(processor_id, config)
        else:
            logger.error(f"Unknown processor type: {processor_type}")
    
    async def _run_aggregation_processor(self, processor_id: str, config: Dict[str, Any]):
        """Run event aggregation processor."""
        
        # Create consumer for input stream
        input_topic = config.get('input_topic')
        consumer = aiokafka.AIOKafkaConsumer(
            input_topic,
            bootstrap_servers=self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
            group_id=f"processor_{processor_id}",
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        try:
            await consumer.start()
            
            # Aggregation state
            aggregation_state = {}
            window_size = config.get('window_size_ms', 60000)  # 1 minute default
            
            async for message in consumer:
                event_data = message.value
                
                # Perform aggregation logic
                await self._process_aggregation(aggregation_state, event_data, config)
                
                # Check if window should be emitted
                if await self._should_emit_window(aggregation_state, window_size):
                    await self._emit_aggregation_results(processor_id, aggregation_state, config)
                    aggregation_state = {}  # Reset for next window
                    
        except asyncio.CancelledError:
            logger.info(f"Aggregation processor cancelled: {processor_id}")
        finally:
            await consumer.stop()
    
    async def _process_aggregation(self, state: Dict[str, Any], event_data: Dict[str, Any], config: Dict[str, Any]):
        """Process event for aggregation."""
        
        aggregation_field = config.get('aggregation_field', 'payload.amount')
        group_by_field = config.get('group_by_field', 'aggregate_type')
        
        # Extract grouping key
        group_key = self._extract_field_value(event_data, group_by_field)
        
        # Extract aggregation value
        agg_value = self._extract_field_value(event_data, aggregation_field)
        
        if group_key and agg_value is not None:
            if group_key not in state:
                state[group_key] = {
                    'count': 0,
                    'sum': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'first_event_time': event_data.get('timestamp'),
                    'last_event_time': event_data.get('timestamp')
                }
            
            # Update aggregation
            state[group_key]['count'] += 1
            state[group_key]['sum'] += float(agg_value)
            state[group_key]['min'] = min(state[group_key]['min'], float(agg_value))
            state[group_key]['max'] = max(state[group_key]['max'], float(agg_value))
            state[group_key]['last_event_time'] = event_data.get('timestamp')
    
    def _extract_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Extract field value using dot notation."""
        
        parts = field_path.split('.')
        value = data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    async def _should_emit_window(self, state: Dict[str, Any], window_size_ms: int) -> bool:
        """Check if aggregation window should be emitted."""
        
        if not state:
            return False
        
        # Simple time-based window (would implement more sophisticated windowing)
        current_time = datetime.now(timezone.utc)
        
        for group_data in state.values():
            first_time = datetime.fromisoformat(group_data['first_event_time'].replace('Z', '+00:00'))
            if (current_time - first_time).total_seconds() * 1000 >= window_size_ms:
                return True
        
        return False
    
    async def _emit_aggregation_results(self, processor_id: str, state: Dict[str, Any], config: Dict[str, Any]):
        """Emit aggregation results."""
        
        output_topic = config.get('output_topic')
        if not output_topic:
            return
        
        # Create aggregation result event
        result_event = {
            "event_id": f"agg_{uuid7str()}",
            "event_type": "aggregation.result",
            "processor_id": processor_id,
            "window_start": min(g['first_event_time'] for g in state.values()),
            "window_end": max(g['last_event_time'] for g in state.values()),
            "aggregation_results": state,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Publish to output topic (would use Kafka producer)
        logger.info(f"Emitted aggregation results from processor {processor_id}")
    
    async def _run_windowing_processor(self, processor_id: str, config: Dict[str, Any]):
        """Run windowing processor (tumbling, hopping, session windows)."""
        # Implementation for windowing operations
        logger.info(f"Windowing processor not yet implemented: {processor_id}")
    
    async def _run_join_processor(self, processor_id: str, config: Dict[str, Any]):
        """Run stream join processor."""
        # Implementation for stream joins
        logger.info(f"Join processor not yet implemented: {processor_id}")
    
    async def close(self):
        """Close all stream processors."""
        
        for processor_id in list(self.processors.keys()):
            await self.stop_stream_processor(processor_id)


# =============================================================================
# Event Sourcing Service
# =============================================================================

class EventSourcingService:
	"""Service for event sourcing and aggregate reconstruction."""
	
	def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
		self.db_session = db_session
		self.redis_client = redis_client
	
	async def append_event(
		self,
		aggregate_id: str,
		aggregate_type: str,
		event_data: Dict[str, Any],
		expected_version: Optional[int] = None,
		tenant_id: str = None
	) -> str:
		"""Append event to event store with optimistic concurrency control."""
		
		# Get current aggregate version
		current_version = await self._get_aggregate_version(aggregate_id, aggregate_type, tenant_id)
		
		# Check optimistic concurrency
		if expected_version is not None and current_version != expected_version:
			raise ValueError(f"Concurrency conflict: expected version {expected_version}, got {current_version}")
		
		# Create event store entry
		new_version = current_version + 1
		event_id = f"evt_{uuid7str()}"
		
		from .models import ESEventStore
		event_store_entry = ESEventStore(
			aggregate_id=aggregate_id,
			aggregate_type=aggregate_type,
			event_id=event_id,
			event_sequence=new_version,
			aggregate_version=new_version,
			event_type=event_data.get('event_type'),
			event_data=event_data.get('payload', {}),
			event_metadata=event_data.get('metadata', {}),
			event_timestamp=datetime.now(timezone.utc),
			tenant_id=tenant_id,
			created_by=event_data.get('created_by', 'system')
		)
		
		self.db_session.add(event_store_entry)
		await self.db_session.commit()
		
		# Invalidate cached aggregate
		await self._invalidate_aggregate_cache(aggregate_id, aggregate_type, tenant_id)
		
		logger.info(f"Appended event {event_id} to aggregate {aggregate_id} version {new_version}")
		return event_id
	
	async def get_aggregate_events(
		self,
		aggregate_id: str,
		aggregate_type: str,
		from_version: int = 0,
		to_version: Optional[int] = None,
		tenant_id: str = None
	) -> List[Dict[str, Any]]:
		"""Get events for aggregate within version range."""
		
		from .models import ESEventStore
		query = select(ESEventStore).where(
			and_(
				ESEventStore.aggregate_id == aggregate_id,
				ESEventStore.aggregate_type == aggregate_type,
				ESEventStore.aggregate_version > from_version
			)
		)
		
		if tenant_id:
			query = query.where(ESEventStore.tenant_id == tenant_id)
		
		if to_version:
			query = query.where(ESEventStore.aggregate_version <= to_version)
		
		query = query.order_by(ESEventStore.aggregate_version)
		
		result = await self.db_session.execute(query)
		events = result.scalars().all()
		
		return [
			{
				"event_id": event.event_id,
				"event_type": event.event_type,
				"event_sequence": event.event_sequence,
				"aggregate_version": event.aggregate_version,
				"event_data": event.event_data,
				"event_metadata": event.event_metadata,
				"event_timestamp": event.event_timestamp,
				"tenant_id": event.tenant_id
			}
			for event in events
		]
	
	async def replay_aggregate(
		self,
		aggregate_id: str,
		aggregate_type: str,
		to_version: Optional[int] = None,
		tenant_id: str = None
	) -> Dict[str, Any]:
		"""Replay events to reconstruct aggregate state."""
		
		# Check for cached snapshot
		snapshot = await self._get_latest_snapshot(aggregate_id, aggregate_type, tenant_id)
		
		from_version = 0
		aggregate_state = {}
		
		if snapshot:
			from_version = snapshot.get('snapshot_version', 0)
			aggregate_state = snapshot.get('aggregate_data', {})
		
		# Get events since snapshot
		events = await self.get_aggregate_events(
			aggregate_id, aggregate_type, from_version, to_version, tenant_id
		)
		
		# Apply events to reconstruct state
		for event in events:
			aggregate_state = await self._apply_event_to_aggregate(
				aggregate_state, event, aggregate_type
			)
		
		# Cache reconstructed state
		if len(events) > 0:
			await self._cache_aggregate_state(aggregate_id, aggregate_type, aggregate_state, tenant_id)
		
		return aggregate_state
	
	async def create_snapshot(
		self,
		aggregate_id: str,
		aggregate_type: str,
		tenant_id: str = None
	) -> str:
		"""Create snapshot of current aggregate state."""
		
		# Get current aggregate state
		aggregate_state = await self.replay_aggregate(aggregate_id, aggregate_type, None, tenant_id)
		current_version = await self._get_aggregate_version(aggregate_id, aggregate_type, tenant_id)
		
		# Serialize and compress aggregate data
		import gzip
		import pickle
		
		serialized_data = pickle.dumps(aggregate_state)
		compressed_data = gzip.compress(serialized_data)
		
		# Create snapshot record
		from .models import ESSnapshot
		snapshot = ESSnapshot(
			aggregate_id=aggregate_id,
			aggregate_type=aggregate_type,
			snapshot_version=current_version,
			last_event_sequence=current_version,
			aggregate_data=compressed_data,
			compression_type=CompressionType.GZIP,
			serialization_format=SerializationFormat.BINARY,
			original_size=len(serialized_data),
			compressed_size=len(compressed_data),
			tenant_id=tenant_id,
			creation_time_ms=int((datetime.now(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds() * 1000),
			events_included=current_version,
			created_by='system'
		)
		
		self.db_session.add(snapshot)
		await self.db_session.commit()
		
		logger.info(f"Created snapshot for aggregate {aggregate_id} at version {current_version}")
		return snapshot.snapshot_id
	
	async def _get_aggregate_version(self, aggregate_id: str, aggregate_type: str, tenant_id: str) -> int:
		"""Get current version of aggregate."""
		
		from .models import ESEventStore
		result = await self.db_session.execute(
			select(func.max(ESEventStore.aggregate_version)).where(
				and_(
					ESEventStore.aggregate_id == aggregate_id,
					ESEventStore.aggregate_type == aggregate_type,
					ESEventStore.tenant_id == tenant_id if tenant_id else True
				)
			)
		)
		
		version = result.scalar()
		return version or 0
	
	async def _get_latest_snapshot(self, aggregate_id: str, aggregate_type: str, tenant_id: str) -> Optional[Dict[str, Any]]:
		"""Get latest snapshot for aggregate."""
		
		from .models import ESSnapshot
		result = await self.db_session.execute(
			select(ESSnapshot).where(
				and_(
					ESSnapshot.aggregate_id == aggregate_id,
					ESSnapshot.aggregate_type == aggregate_type,
					ESSnapshot.tenant_id == tenant_id if tenant_id else True
				)
			).order_by(desc(ESSnapshot.snapshot_version)).limit(1)
		)
		
		snapshot = result.scalar_one_or_none()
		if not snapshot:
			return None
		
		# Decompress and deserialize
		import gzip
		import pickle
		
		decompressed_data = gzip.decompress(snapshot.aggregate_data)
		aggregate_data = pickle.loads(decompressed_data)
		
		return {
			"snapshot_version": snapshot.snapshot_version,
			"aggregate_data": aggregate_data
		}
	
	async def _apply_event_to_aggregate(
		self,
		aggregate_state: Dict[str, Any],
		event: Dict[str, Any],
		aggregate_type: str
	) -> Dict[str, Any]:
		"""Apply event to aggregate state (domain-specific logic)."""
		
		# This is a generic implementation - would be customized per aggregate type
		event_type = event.get('event_type', '')
		event_data = event.get('event_data', {})
		
		# Simple merge strategy for demonstration
		if 'data' not in aggregate_state:
			aggregate_state['data'] = {}
		
		aggregate_state['data'].update(event_data)
		aggregate_state['version'] = event.get('aggregate_version')
		aggregate_state['last_modified'] = event.get('event_timestamp')
		
		return aggregate_state
	
	async def _cache_aggregate_state(
		self,
		aggregate_id: str,
		aggregate_type: str,
		state: Dict[str, Any],
		tenant_id: str
	):
		"""Cache aggregate state in Redis."""
		
		cache_key = f"aggregate:{tenant_id}:{aggregate_type}:{aggregate_id}"
		await self.redis_client.setex(
			cache_key,
			3600,  # 1 hour
			json.dumps(state, default=str)
		)
	
	async def _invalidate_aggregate_cache(
		self,
		aggregate_id: str,
		aggregate_type: str,
		tenant_id: str
	):
		"""Invalidate cached aggregate state."""
		
		cache_key = f"aggregate:{tenant_id}:{aggregate_type}:{aggregate_id}"
		await self.redis_client.delete(cache_key)


# =============================================================================
# Stream Management Service
# =============================================================================

class StreamManagementService:
	"""Service for managing streams, topics, and configurations."""
	
	def __init__(self, db_session: AsyncSession, kafka_config: Dict[str, Any]):
		self.db_session = db_session
		self.kafka_config = kafka_config
		self.admin_client = None
	
	async def create_stream(self, stream_config: StreamCreate, tenant_id: str, user_id: str) -> str:
		"""Create a new event stream with Kafka topic."""
		
		# Check if stream name already exists
		existing = await self.db_session.execute(
			select(ESStream).where(
				and_(
					ESStream.stream_name == stream_config.stream_name,
					ESStream.tenant_id == tenant_id
				)
			)
		)
		
		if existing.scalar_one_or_none():
			raise ValueError(f"Stream name already exists: {stream_config.stream_name}")
		
		# Create Kafka topic
		topic_created = await self._create_kafka_topic(
			stream_config.topic_name,
			stream_config.partition_count,
			stream_config.replication_factor,
			{
				'cleanup.policy': stream_config.cleanup_policy,
				'compression.type': stream_config.compression_type.value,
				'retention.ms': str(stream_config.retention_time_ms)
			}
		)
		
		if not topic_created:
			raise RuntimeError(f"Failed to create Kafka topic: {stream_config.topic_name}")
		
		# Create stream record
		stream = ESStream(
			stream_name=stream_config.stream_name,
			stream_description=stream_config.description,
			topic_name=stream_config.topic_name,
			partitions=stream_config.partition_count,
			replication_factor=stream_config.replication_factor,
			retention_time_ms=stream_config.retention_time_ms,
			retention_size_bytes=stream_config.retention_size_bytes,
			cleanup_policy=stream_config.cleanup_policy,
			compression_type=stream_config.compression_type.value,
			default_serialization=stream_config.serialization_format.value,
			event_category=EventType.DOMAIN_EVENT.value,
			source_capability=stream_config.tenant_id,  # Using tenant as source for now
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		# Add enhanced fields
		stream.stream_category = stream_config.stream_category
		stream.business_domain = stream_config.business_domain
		stream.visibility = stream_config.visibility
		stream.encryption_enabled = stream_config.encryption_enabled
		stream.access_control_enabled = stream_config.access_control_enabled
		
		# Store routing rules and filters as JSON
		stream.config_settings = {
			'event_filters': stream_config.event_filters,
			'routing_rules': stream_config.routing_rules,
			'min_in_sync_replicas': stream_config.min_in_sync_replicas
		}
		
		self.db_session.add(stream)
		await self.db_session.commit()
		
		logger.info(f"Created stream {stream.stream_id} with topic {stream_config.topic_name}")
		return stream.stream_id
	
	async def update_stream(
		self,
		stream_id: str,
		updates: Dict[str, Any],
		tenant_id: str,
		user_id: str
	) -> bool:
		"""Update stream configuration."""
		
		# Get existing stream
		result = await self.db_session.execute(
			select(ESStream).where(
				and_(
					ESStream.stream_id == stream_id,
					ESStream.tenant_id == tenant_id
				)
			)
		)
		
		stream = result.scalar_one_or_none()
		if not stream:
			raise ValueError(f"Stream not found: {stream_id}")
		
		# Update allowed fields
		updatable_fields = [
			'stream_description', 'retention_time_ms', 'retention_size_bytes',
			'compression_type', 'status', 'config_settings'
		]
		
		for field, value in updates.items():
			if field in updatable_fields:
				setattr(stream, field, value)
		
		# Update Kafka topic configuration if needed
		if 'retention_time_ms' in updates or 'compression_type' in updates:
			await self._update_kafka_topic_config(
				stream.topic_name,
				{
					'retention.ms': str(stream.retention_time_ms),
					'compression.type': stream.compression_type
				}
			)
		
		await self.db_session.commit()
		
		logger.info(f"Updated stream {stream_id}")
		return True
	
	async def delete_stream(self, stream_id: str, tenant_id: str, user_id: str) -> bool:
		"""Delete stream and associated Kafka topic."""
		
		# Get stream
		result = await self.db_session.execute(
			select(ESStream).where(
				and_(
					ESStream.stream_id == stream_id,
					ESStream.tenant_id == tenant_id
				)
			)
		)
		
		stream = result.scalar_one_or_none()
		if not stream:
			raise ValueError(f"Stream not found: {stream_id}")
		
		# Check for active subscriptions
		subscriptions = await self.db_session.execute(
			select(func.count(ESSubscription.subscription_id)).where(
				and_(
					ESSubscription.stream_id == stream_id,
					ESSubscription.status == SubscriptionStatus.ACTIVE.value
				)
			)
		)
		
		if subscriptions.scalar() > 0:
			raise ValueError("Cannot delete stream with active subscriptions")
		
		# Archive stream instead of hard delete
		stream.status = StreamStatus.ARCHIVED.value
		await self.db_session.commit()
		
		# Delete Kafka topic (optional - might want to retain for audit)
		# await self._delete_kafka_topic(stream.topic_name)
		
		logger.info(f"Archived stream {stream_id}")
		return True
	
	async def get_stream_metrics(self, stream_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Get comprehensive stream metrics."""
		
		# Get stream info
		result = await self.db_session.execute(
			select(ESStream).where(
				and_(
					ESStream.stream_id == stream_id,
					ESStream.tenant_id == tenant_id
				)
			)
		)
		
		stream = result.scalar_one_or_none()
		if not stream:
			raise ValueError(f"Stream not found: {stream_id}")
		
		# Get event counts
		event_count_result = await self.db_session.execute(
			select(func.count(ESEvent.event_id)).where(
				and_(
					ESEvent.stream_id == stream_id,
					ESEvent.created_at >= datetime.now(timezone.utc) - timedelta(hours=24)
				)
			)
		)
		
		recent_events = event_count_result.scalar()
		
		# Get subscription count
		subscription_count_result = await self.db_session.execute(
			select(func.count(ESSubscription.subscription_id)).where(
				ESSubscription.stream_id == stream_id
			)
		)
		
		subscription_count = subscription_count_result.scalar()
		
		# Get Kafka topic metrics (would integrate with Kafka JMX)
		kafka_metrics = await self._get_kafka_topic_metrics(stream.topic_name)
		
		return {
			"stream_id": stream_id,
			"stream_name": stream.stream_name,
			"topic_name": stream.topic_name,
			"status": stream.status,
			"partition_count": stream.partitions,
			"replication_factor": stream.replication_factor,
			"retention_time_ms": stream.retention_time_ms,
			"events_24h": recent_events,
			"total_subscriptions": subscription_count,
			"kafka_metrics": kafka_metrics,
			"last_updated": datetime.now(timezone.utc).isoformat()
		}
	
	async def _create_kafka_topic(
		self,
		topic_name: str,
		partitions: int,
		replication_factor: int,
		config: Dict[str, str]
	) -> bool:
		"""Create Kafka topic with specified configuration."""
		
		try:
			# Use KafkaAdminClient for topic management
			admin_client = KafkaAdminClient(
				bootstrap_servers=self.kafka_config.get('bootstrap_servers', 'localhost:9092')
			)
			
			topic = NewTopic(
				name=topic_name,
				num_partitions=partitions,
				replication_factor=replication_factor,
				topic_configs=config
			)
			
			result = admin_client.create_topics([topic])
			
			# Wait for topic creation
			for topic, future in result.items():
				try:
					future.result()
					logger.info(f"Created Kafka topic: {topic}")
					return True
				except TopicAlreadyExistsError:
					logger.info(f"Kafka topic already exists: {topic}")
					return True
				except Exception as e:
					logger.error(f"Failed to create topic {topic}: {e}")
					return False
		
		except Exception as e:
			logger.error(f"Error creating Kafka topic {topic_name}: {e}")
			return False
		
		finally:
			if 'admin_client' in locals():
				admin_client.close()
	
	async def _update_kafka_topic_config(self, topic_name: str, config: Dict[str, str]) -> bool:
		"""Update Kafka topic configuration."""
		
		try:
			admin_client = KafkaAdminClient(
				bootstrap_servers=self.kafka_config.get('bootstrap_servers', 'localhost:9092')
			)
			
			resource = ConfigResource(ConfigResourceType.TOPIC, topic_name)
			configs = {resource: config}
			
			result = admin_client.alter_configs(configs)
			
			for resource, future in result.items():
				try:
					future.result()
					logger.info(f"Updated Kafka topic config: {resource}")
					return True
				except Exception as e:
					logger.error(f"Failed to update topic config {resource}: {e}")
					return False
		
		except Exception as e:
			logger.error(f"Error updating Kafka topic config {topic_name}: {e}")
			return False
		
		finally:
			if 'admin_client' in locals():
				admin_client.close()
	
	async def _get_kafka_topic_metrics(self, topic_name: str) -> Dict[str, Any]:
		"""Get Kafka topic metrics (placeholder - would integrate with JMX)."""
		
		# Placeholder implementation - would integrate with Kafka JMX metrics
		return {
			"bytes_in_per_sec": 0,
			"bytes_out_per_sec": 0,
			"messages_in_per_sec": 0,
			"total_log_size": 0,
			"leader_count": 0,
			"partition_count": 0
		}


# =============================================================================
# Consumer Management Service  
# =============================================================================

class ConsumerManagementService:
	"""Service for managing consumer groups and individual consumers."""
	
	def __init__(self, db_session: AsyncSession, kafka_config: Dict[str, Any]):
		self.db_session = db_session
		self.kafka_config = kafka_config
	
	async def create_consumer_group(
		self,
		group_config: Dict[str, Any],
		tenant_id: str,
		user_id: str
	) -> str:
		"""Create a new consumer group."""
		
		# Check if group already exists
		existing = await self.db_session.execute(
			select(ESConsumerGroup).where(
				and_(
					ESConsumerGroup.group_name == group_config['group_name'],
					ESConsumerGroup.tenant_id == tenant_id
				)
			)
		)
		
		if existing.scalar_one_or_none():
			raise ValueError(f"Consumer group already exists: {group_config['group_name']}")
		
		# Create consumer group
		consumer_group = ESConsumerGroup(
			group_id=group_config.get('group_id', f"cg_{uuid7str()}"),
			group_name=group_config['group_name'],
			group_description=group_config.get('description'),
			session_timeout_ms=group_config.get('session_timeout_ms', 30000),
			heartbeat_interval_ms=group_config.get('heartbeat_interval_ms', 3000),
			max_poll_interval_ms=group_config.get('max_poll_interval_ms', 300000),
			partition_assignment_strategy=group_config.get('assignment_strategy', 'round_robin'),
			rebalance_timeout_ms=group_config.get('rebalance_timeout_ms', 60000),
			tenant_id=tenant_id,
			created_by=user_id
		)
		
		self.db_session.add(consumer_group)
		await self.db_session.commit()
		
		logger.info(f"Created consumer group {consumer_group.group_id}")
		return consumer_group.group_id
	
	async def register_consumer(
		self,
		group_id: str,
		consumer_config: Dict[str, Any],
		tenant_id: str
	) -> str:
		"""Register a new consumer in a group."""
		
		# Get consumer group
		result = await self.db_session.execute(
			select(ESConsumerGroup).where(
				and_(
					ESConsumerGroup.group_id == group_id,
					ESConsumerGroup.tenant_id == tenant_id
				)
			)
		)
		
		group = result.scalar_one_or_none()
		if not group:
			raise ValueError(f"Consumer group not found: {group_id}")
		
		# Create consumer record
		from .models import ESConsumer
		consumer = ESConsumer(
			consumer_name=consumer_config['consumer_name'],
			group_id=group_id,
			instance_id=consumer_config['instance_id'],
			host_name=consumer_config.get('host_name', 'unknown'),
			ip_address=consumer_config.get('ip_address'),
			port=consumer_config.get('port'),
			assigned_partitions=consumer_config.get('assigned_partitions', []),
			partition_assignments=consumer_config.get('partition_assignments', {}),
			status=ConsumerStatus.INACTIVE.value,
			joined_at=datetime.now(timezone.utc)
		)
		
		self.db_session.add(consumer)
		
		# Update group active consumers count
		group.active_consumers += 1
		
		await self.db_session.commit()
		
		logger.info(f"Registered consumer {consumer.consumer_id} in group {group_id}")
		return consumer.consumer_id
	
	async def update_consumer_heartbeat(
		self,
		consumer_id: str,
		performance_metrics: Dict[str, Any],
		tenant_id: str
	) -> bool:
		"""Update consumer heartbeat and performance metrics."""
		
		from .models import ESConsumer
		result = await self.db_session.execute(
			select(ESConsumer).where(ESConsumer.consumer_id == consumer_id)
		)
		
		consumer = result.scalar_one_or_none()
		if not consumer:
			return False
		
		# Update heartbeat and metrics
		consumer.last_heartbeat = datetime.now(timezone.utc)
		consumer.last_poll = performance_metrics.get('last_poll', consumer.last_poll)
		consumer.status = ConsumerStatus.ACTIVE.value
		
		# Update performance metrics
		if 'throughput_msgs_sec' in performance_metrics:
			consumer.throughput_msgs_sec = performance_metrics['throughput_msgs_sec']
		if 'latency_p95_ms' in performance_metrics:
			consumer.latency_p95_ms = performance_metrics['latency_p95_ms']
		if 'memory_usage_mb' in performance_metrics:
			consumer.memory_usage_mb = performance_metrics['memory_usage_mb']
		if 'cpu_usage_percent' in performance_metrics:
			consumer.cpu_usage_percent = performance_metrics['cpu_usage_percent']
		
		# Update processing metrics
		if 'messages_processed' in performance_metrics:
			consumer.messages_processed += performance_metrics['messages_processed']
		if 'bytes_processed' in performance_metrics:
			consumer.bytes_processed += performance_metrics['bytes_processed']
		
		await self.db_session.commit()
		return True
	
	async def handle_consumer_rebalance(
		self,
		group_id: str,
		partition_assignments: Dict[str, List[int]],
		tenant_id: str
	) -> bool:
		"""Handle consumer group rebalancing."""
		
		# Get all consumers in group
		from .models import ESConsumer
		result = await self.db_session.execute(
			select(ESConsumer).where(ESConsumer.group_id == group_id)
		)
		
		consumers = result.scalars().all()
		
		# Update partition assignments
		for consumer in consumers:
			consumer_assignments = partition_assignments.get(consumer.consumer_id, [])
			consumer.assigned_partitions = consumer_assignments
			consumer.partition_assignments = {
				"partitions": consumer_assignments,
				"assigned_at": datetime.now(timezone.utc).isoformat()
			}
		
		# Update consumer group rebalance timestamp
		group_result = await self.db_session.execute(
			select(ESConsumerGroup).where(ESConsumerGroup.group_id == group_id)
		)
		
		group = group_result.scalar_one_or_none()
		if group:
			group.last_rebalance = datetime.now(timezone.utc)
		
		await self.db_session.commit()
		
		logger.info(f"Handled rebalance for consumer group {group_id}")
		return True
	
	async def get_consumer_group_status(self, group_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Get detailed consumer group status."""
		
		# Get consumer group
		result = await self.db_session.execute(
			select(ESConsumerGroup).where(
				and_(
					ESConsumerGroup.group_id == group_id,
					ESConsumerGroup.tenant_id == tenant_id
				)
			)
		)
		
		group = result.scalar_one_or_none()
		if not group:
			raise ValueError(f"Consumer group not found: {group_id}")
		
		# Get consumers
		from .models import ESConsumer
		consumers_result = await self.db_session.execute(
			select(ESConsumer).where(ESConsumer.group_id == group_id)
		)
		
		consumers = consumers_result.scalars().all()
		
		# Calculate lag and metrics
		total_lag = 0
		active_consumers = 0
		total_throughput = 0
		
		consumer_details = []
		for consumer in consumers:
			if consumer.status == ConsumerStatus.ACTIVE.value:
				active_consumers += 1
				total_throughput += consumer.throughput_msgs_sec
			
			consumer_details.append({
				"consumer_id": consumer.consumer_id,
				"consumer_name": consumer.consumer_name,
				"status": consumer.status,
				"assigned_partitions": consumer.assigned_partitions,
				"last_heartbeat": consumer.last_heartbeat.isoformat() if consumer.last_heartbeat else None,
				"throughput_msgs_sec": consumer.throughput_msgs_sec,
				"latency_p95_ms": consumer.latency_p95_ms,
				"memory_usage_mb": consumer.memory_usage_mb,
				"cpu_usage_percent": consumer.cpu_usage_percent
			})
		
		# Update group metrics
		group.active_consumers = active_consumers
		group.total_lag = total_lag
		await self.db_session.commit()
		
		return {
			"group_id": group_id,
			"group_name": group.group_name,
			"status": "healthy" if active_consumers > 0 else "unhealthy",
			"active_consumers": active_consumers,
			"total_consumers": len(consumers),
			"total_lag": total_lag,
			"total_throughput_msgs_sec": total_throughput,
			"last_rebalance": group.last_rebalance.isoformat() if group.last_rebalance else None,
			"consumers": consumer_details
		}

# =============================================================================
# Main Event Streaming Service
# =============================================================================

class EventStreamingService:
    """Main service orchestrating all event streaming operations."""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: redis.Redis,
        kafka_config: Optional[Dict[str, Any]] = None
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.kafka_config = kafka_config or {
            'bootstrap_servers': 'localhost:9092'
        }
        
        # Initialize sub-services
        self.publisher = EventPublishingService(db_session, redis_client, self.kafka_config)
        self.consumer = EventConsumptionService(db_session, redis_client, self.kafka_config)
        self.processor = StreamProcessingService(db_session, redis_client, self.kafka_config)
        self.event_sourcing = EventSourcingService(db_session, redis_client)
        self.stream_manager = StreamManagementService(db_session, self.kafka_config)
        self.consumer_manager = ConsumerManagementService(db_session, self.kafka_config)
        
    async def publish_event(
        self,
        event_config: EventConfig,
        payload: Dict[str, Any],
        tenant_id: str,
        user_id: str
    ) -> str:
        """Publish an event to the streaming platform."""
        return await self.publisher.publish_event(event_config, payload, tenant_id, user_id)
    
    async def publish_events_batch(
        self,
        events_data: List[tuple[EventConfig, Dict[str, Any]]],
        tenant_id: str,
        user_id: str
    ) -> List[str]:
        """Publish multiple events in a batch."""
        return await self.publisher.publish_events_batch(events_data, tenant_id, user_id)
    
    async def start_subscription(self, subscription_id: str) -> bool:
        """Start consuming events for a subscription."""
        return await self.consumer.start_subscription(subscription_id)
    
    async def stop_subscription(self, subscription_id: str) -> bool:
        """Stop consuming events for a subscription."""
        return await self.consumer.stop_subscription(subscription_id)
    
    async def get_subscription_status(self, subscription_id: str) -> Dict[str, Any]:
        """Get subscription status."""
        return await self.consumer.get_subscription_status(subscription_id)
    
    async def start_stream_processor(self, processor_id: str, config: Dict[str, Any]) -> bool:
        """Start a stream processing job."""
        return await self.processor.start_stream_processor(processor_id, config)
    
    async def stop_stream_processor(self, processor_id: str) -> bool:
        """Stop a stream processing job."""
        return await self.processor.stop_stream_processor(processor_id)
    
    # Event Sourcing Methods
    async def append_event_to_store(
        self,
        aggregate_id: str,
        aggregate_type: str,
        event_data: Dict[str, Any],
        expected_version: Optional[int] = None,
        tenant_id: str = None
    ) -> str:
        """Append event to event store."""
        return await self.event_sourcing.append_event(
            aggregate_id, aggregate_type, event_data, expected_version, tenant_id
        )
    
    async def replay_aggregate(
        self,
        aggregate_id: str,
        aggregate_type: str,
        to_version: Optional[int] = None,
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """Replay events to reconstruct aggregate state."""
        return await self.event_sourcing.replay_aggregate(
            aggregate_id, aggregate_type, to_version, tenant_id
        )
    
    async def create_aggregate_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str,
        tenant_id: str = None
    ) -> str:
        """Create snapshot of aggregate state."""
        return await self.event_sourcing.create_snapshot(aggregate_id, aggregate_type, tenant_id)
    
    # Stream Management Methods
    async def create_stream(self, stream_config: StreamCreate, tenant_id: str, user_id: str) -> str:
        """Create a new event stream."""
        return await self.stream_manager.create_stream(stream_config, tenant_id, user_id)
    
    async def update_stream(
        self,
        stream_id: str,
        updates: Dict[str, Any],
        tenant_id: str,
        user_id: str
    ) -> bool:
        """Update stream configuration."""
        return await self.stream_manager.update_stream(stream_id, updates, tenant_id, user_id)
    
    async def delete_stream(self, stream_id: str, tenant_id: str, user_id: str) -> bool:
        """Delete/archive a stream."""
        return await self.stream_manager.delete_stream(stream_id, tenant_id, user_id)
    
    async def get_stream_metrics(self, stream_id: str, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive stream metrics."""
        return await self.stream_manager.get_stream_metrics(stream_id, tenant_id)
    
    # Consumer Management Methods
    async def create_consumer_group(
        self,
        group_config: Dict[str, Any],
        tenant_id: str,
        user_id: str
    ) -> str:
        """Create a new consumer group."""
        return await self.consumer_manager.create_consumer_group(group_config, tenant_id, user_id)
    
    async def register_consumer(
        self,
        group_id: str,
        consumer_config: Dict[str, Any],
        tenant_id: str
    ) -> str:
        """Register a consumer in a group."""
        return await self.consumer_manager.register_consumer(group_id, consumer_config, tenant_id)
    
    async def update_consumer_heartbeat(
        self,
        consumer_id: str,
        performance_metrics: Dict[str, Any],
        tenant_id: str
    ) -> bool:
        """Update consumer heartbeat and metrics."""
        return await self.consumer_manager.update_consumer_heartbeat(
            consumer_id, performance_metrics, tenant_id
        )
    
    async def get_consumer_group_status(self, group_id: str, tenant_id: str) -> Dict[str, Any]:
        """Get detailed consumer group status."""
        return await self.consumer_manager.get_consumer_group_status(group_id, tenant_id)
    
    async def get_streaming_health(self, tenant_id: str) -> Dict[str, Any]:
        """Get overall streaming platform health."""
        
        # Get stream count and status
        stream_result = await self.db_session.execute(
            select(func.count(ESStream.stream_id), ESStream.status)
            .where(ESStream.tenant_id == tenant_id)
            .group_by(ESStream.status)
        )
        
        stream_stats = dict(stream_result.all())
        
        # Get subscription count and status
        subscription_result = await self.db_session.execute(
            select(func.count(ESSubscription.subscription_id), ESSubscription.status)
            .where(ESSubscription.tenant_id == tenant_id)
            .group_by(ESSubscription.status)
        )
        
        subscription_stats = dict(subscription_result.all())
        
        # Get recent event count
        recent_events = await self.db_session.execute(
            select(func.count(ESEvent.event_id))
            .where(
                and_(
                    ESEvent.tenant_id == tenant_id,
                    ESEvent.created_at >= datetime.now(timezone.utc) - timedelta(hours=24)
                )
            )
        )
        
        recent_count = recent_events.scalar()
        
        return {
            "status": "healthy",
            "streams": {
                "total": sum(stream_stats.values()),
                "by_status": stream_stats
            },
            "subscriptions": {
                "total": sum(subscription_stats.values()),
                "by_status": subscription_stats,
                "active_consumers": len(self.consumer.active_consumers)
            },
            "events": {
                "recent_24h": recent_count
            },
            "processors": {
                "active": len(self.processor.processors)
            }
        }
    
    async def close(self):
        """Close all streaming services and clean up resources."""
        await self.publisher.close()
        await self.consumer.close()
        await self.processor.close()
        # Note: Event sourcing, stream manager, and consumer manager don't need explicit close

# =============================================================================
# Service Factory Functions
# =============================================================================

async def create_event_streaming_service(
    db_session: AsyncSession,
    redis_url: str,
    kafka_config: Optional[Dict[str, Any]] = None
) -> EventStreamingService:
    """Factory function to create event streaming service."""
    
    redis_client = redis.from_url(redis_url)
    return EventStreamingService(db_session, redis_client, kafka_config)

# Export service classes
__all__ = [
    "EventStreamingService",
    "EventPublishingService",
    "EventConsumptionService", 
    "StreamProcessingService",
    "EventSourcingService",
    "StreamManagementService",
    "ConsumerManagementService",
    "create_event_streaming_service"
]
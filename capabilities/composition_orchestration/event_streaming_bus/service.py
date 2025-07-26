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
    EventStatus, StreamStatus, SubscriptionStatus, EventType, DeliveryMode,
    EventConfig, StreamConfig, SubscriptionConfig, SchemaConfig
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
    "create_event_streaming_service"
]
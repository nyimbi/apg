"""
APG Event Streaming Bus - REST API Layer

Comprehensive REST API with WebSocket support for real-time event streaming,
subscription management, and stream operations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND

from .models import (
	ESEvent, ESStream, ESSubscription, ESConsumerGroup, ESSchema, ESMetrics,
	ESEventSchema, ESStreamAssignment, ESEventProcessingHistory, ESStreamProcessor,
	EventConfig, StreamConfig, SubscriptionConfig, SchemaConfig,
	EventStatus, StreamStatus, SubscriptionStatus, EventType, EventPriority,
	ConsumerStatus, ProcessorType, CompressionType, SerializationFormat
)
from .service import (
	EventStreamingService, EventPublishingService, EventConsumptionService,
	StreamProcessingService, EventSourcingService, SchemaRegistryService,
	StreamManagementService, ConsumerManagementService
)

# =============================================================================
# API Models
# =============================================================================

class EventPublishRequest(BaseModel):
	"""Request model for publishing events."""
	event_type: str = Field(..., min_length=1, max_length=100)
	payload: Dict[str, Any] = Field(...)
	source_capability: str = Field(..., min_length=1, max_length=100)
	aggregate_id: str = Field(..., min_length=1, max_length=100)
	aggregate_type: str = Field(..., min_length=1, max_length=100)
	stream_id: Optional[str] = Field(None, max_length=100)
	correlation_id: Optional[str] = Field(None, max_length=100)
	causation_id: Optional[str] = Field(None, max_length=100)
	metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
	partition_key: Optional[str] = Field(None, max_length=200)
	# Enhanced fields
	priority: Optional[EventPriority] = Field(default=EventPriority.NORMAL)
	compression_type: Optional[CompressionType] = Field(default=CompressionType.NONE)
	serialization_format: Optional[SerializationFormat] = Field(default=SerializationFormat.JSON)
	schema_id: Optional[str] = Field(None, max_length=100)
	max_retries: Optional[int] = Field(default=3, ge=0, le=10)

class EventBatchPublishRequest(BaseModel):
	"""Request model for batch event publishing."""
	events: List[EventPublishRequest] = Field(..., min_items=1, max_items=1000)
	stream_id: Optional[str] = Field(None, max_length=100)
	batch_options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class EventQueryRequest(BaseModel):
	"""Request model for querying events."""
	stream_id: Optional[str] = None
	event_type: Optional[str] = None
	source_capability: Optional[str] = None
	aggregate_id: Optional[str] = None
	aggregate_type: Optional[str] = None
	correlation_id: Optional[str] = None
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	status: Optional[EventStatus] = None
	limit: int = Field(default=100, ge=1, le=10000)
	offset: int = Field(default=0, ge=0)

class EventResponse(BaseModel):
	"""Response model for event data."""
	event_id: str
	event_type: str
	event_version: str
	source_capability: str
	aggregate_id: str
	aggregate_type: str
	sequence_number: int
	timestamp: datetime
	correlation_id: Optional[str]
	causation_id: Optional[str]
	payload: Dict[str, Any]
	metadata: Dict[str, Any]
	status: EventStatus
	stream_id: str
	# Enhanced fields
	priority: EventPriority
	compression_type: CompressionType
	serialization_format: SerializationFormat
	schema_id: Optional[str]
	processing_duration_ms: Optional[int]
	retry_count: int
	max_retries: int
	error_message: Optional[str]

class StreamMetricsResponse(BaseModel):
	"""Response model for stream metrics."""
	stream_id: str
	stream_name: str
	total_events: int
	events_per_second: float
	events_today: int
	events_last_hour: int
	consumer_count: int
	total_lag: int
	health_status: str
	last_event_time: Optional[datetime]

class SubscriptionStatusResponse(BaseModel):
	"""Response model for subscription status."""
	subscription_id: str
	subscription_name: str
	status: SubscriptionStatus
	consumer_lag: int
	last_consumed_offset: Optional[int]
	last_consumed_at: Optional[datetime]
	events_processed_today: int
	processing_rate: float
	error_count: int

class WebSocketMessage(BaseModel):
	"""WebSocket message format."""
	message_type: str = Field(..., description="Type of message: event, status, error, ping")
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	data: Dict[str, Any] = Field(default_factory=dict)
	correlation_id: Optional[str] = None

# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
	title="APG Event Streaming Bus API",
	description="Enterprise-grade event streaming platform API",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Configure appropriately for production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

# Service instances (would be injected in production)
event_streaming_service = None
event_publishing_service = None
event_consumption_service = None
stream_processing_service = None
schema_registry_service = None
event_sourcing_service = None
stream_management_service = None
consumer_management_service = None

# WebSocket connection manager
class ConnectionManager:
	"""Manage WebSocket connections for real-time streaming."""
	
	def __init__(self):
		self.active_connections: Dict[str, WebSocket] = {}
		self.stream_subscribers: Dict[str, List[str]] = {}
		self.subscription_subscribers: Dict[str, List[str]] = {}
	
	async def connect(self, websocket: WebSocket, connection_id: str):
		"""Accept and store WebSocket connection."""
		await websocket.accept()
		self.active_connections[connection_id] = websocket
	
	def disconnect(self, connection_id: str):
		"""Remove WebSocket connection."""
		if connection_id in self.active_connections:
			del self.active_connections[connection_id]
		
		# Remove from all subscriptions
		for stream_id, subscribers in self.stream_subscribers.items():
			if connection_id in subscribers:
				subscribers.remove(connection_id)
		
		for sub_id, subscribers in self.subscription_subscribers.items():
			if connection_id in subscribers:
				subscribers.remove(connection_id)
	
	def subscribe_to_stream(self, connection_id: str, stream_id: str):
		"""Subscribe connection to stream events."""
		if stream_id not in self.stream_subscribers:
			self.stream_subscribers[stream_id] = []
		if connection_id not in self.stream_subscribers[stream_id]:
			self.stream_subscribers[stream_id].append(connection_id)
	
	def subscribe_to_subscription(self, connection_id: str, subscription_id: str):
		"""Subscribe connection to subscription updates."""
		if subscription_id not in self.subscription_subscribers:
			self.subscription_subscribers[subscription_id] = []
		if connection_id not in self.subscription_subscribers[subscription_id]:
			self.subscription_subscribers[subscription_id].append(connection_id)
	
	async def send_to_connection(self, connection_id: str, message: WebSocketMessage):
		"""Send message to specific connection."""
		if connection_id in self.active_connections:
			try:
				await self.active_connections[connection_id].send_text(message.model_dump_json())
			except:
				# Connection closed, remove it
				self.disconnect(connection_id)
	
	async def broadcast_to_stream(self, stream_id: str, message: WebSocketMessage):
		"""Broadcast message to all stream subscribers."""
		if stream_id in self.stream_subscribers:
			disconnected = []
			for connection_id in self.stream_subscribers[stream_id]:
				try:
					await self.send_to_connection(connection_id, message)
				except:
					disconnected.append(connection_id)
			
			# Clean up disconnected clients
			for connection_id in disconnected:
				self.disconnect(connection_id)
	
	async def broadcast_to_subscription(self, subscription_id: str, message: WebSocketMessage):
		"""Broadcast message to all subscription subscribers."""
		if subscription_id in self.subscription_subscribers:
			disconnected = []
			for connection_id in self.subscription_subscribers[subscription_id]:
				try:
					await self.send_to_connection(connection_id, message)
				except:
					disconnected.append(connection_id)
			
			# Clean up disconnected clients
			for connection_id in disconnected:
				self.disconnect(connection_id)

connection_manager = ConnectionManager()

# =============================================================================
# Dependency Injection
# =============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
	"""Extract user information from JWT token."""
	# In production, validate JWT token and extract user info
	return {"user_id": "api_user", "tenant_id": "default_tenant"}

async def get_event_streaming_service():
	"""Get event streaming service instance."""
	global event_streaming_service
	if not event_streaming_service:
		event_streaming_service = EventStreamingService()
	return event_streaming_service

async def get_event_publishing_service():
	"""Get event publishing service instance."""
	global event_publishing_service
	if not event_publishing_service:
		event_publishing_service = EventPublishingService()
	return event_publishing_service

async def get_event_consumption_service():
	"""Get event consumption service instance."""
	global event_consumption_service
	if not event_consumption_service:
		event_consumption_service = EventConsumptionService()
	return event_consumption_service

async def get_schema_registry_service():
	"""Get schema registry service instance."""
	global schema_registry_service
	if not schema_registry_service:
		schema_registry_service = SchemaRegistryService()
	return schema_registry_service

async def get_event_sourcing_service():
	"""Get event sourcing service instance."""
	global event_sourcing_service
	if not event_sourcing_service:
		event_sourcing_service = EventSourcingService()
	return event_sourcing_service

async def get_stream_management_service():
	"""Get stream management service instance."""
	global stream_management_service
	if not stream_management_service:
		stream_management_service = StreamManagementService()
	return stream_management_service

async def get_consumer_management_service():
	"""Get consumer management service instance."""
	global consumer_management_service
	if not consumer_management_service:
		consumer_management_service = ConsumerManagementService()
	return consumer_management_service

# =============================================================================
# Event Publishing Endpoints
# =============================================================================

@app.post("/api/v1/events", status_code=HTTP_201_CREATED, response_model=EventResponse)
async def publish_event(
	request: EventPublishRequest,
	user: dict = Depends(get_current_user),
	publishing_service: EventPublishingService = Depends(get_event_publishing_service)
):
	"""Publish a single event to the streaming platform."""
	try:
		# Create event configuration
		event_config = EventConfig(
			event_type=request.event_type,
			source_capability=request.source_capability,
			aggregate_id=request.aggregate_id,
			aggregate_type=request.aggregate_type,
			correlation_id=request.correlation_id,
			causation_id=request.causation_id,
			partition_key=request.partition_key,
			metadata=request.metadata,
			# Enhanced configuration
			priority=request.priority.value if request.priority else EventPriority.NORMAL.value,
			compression_type=request.compression_type.value if request.compression_type else CompressionType.NONE.value,
			serialization_format=request.serialization_format.value if request.serialization_format else SerializationFormat.JSON.value,
			schema_id=request.schema_id,
			max_retries=request.max_retries or 3
		)
		
		# Publish event
		event_id = await publishing_service.publish_event(
			event_config=event_config,
			payload=request.payload,
			stream_id=request.stream_id,
			tenant_id=user["tenant_id"],
			user_id=user["user_id"]
		)
		
		# Get published event for response
		event = await publishing_service.get_event(event_id)
		
		# Broadcast to WebSocket subscribers
		if event and event.stream_id:
			message = WebSocketMessage(
				message_type="event",
				data={
					"event_id": event.event_id,
					"event_type": event.event_type,
					"stream_id": event.stream_id,
					"payload": event.payload,
					"priority": event.priority
				}
			)
			await connection_manager.broadcast_to_stream(event.stream_id, message)
		
		return EventResponse(
			event_id=event.event_id,
			event_type=event.event_type,
			event_version=event.event_version,
			source_capability=event.source_capability,
			aggregate_id=event.aggregate_id,
			aggregate_type=event.aggregate_type,
			sequence_number=event.sequence_number,
			timestamp=event.timestamp,
			correlation_id=event.correlation_id,
			causation_id=event.causation_id,
			payload=event.payload,
			metadata=event.metadata,
			status=EventStatus(event.status),
			stream_id=event.stream_id,
			# Enhanced response fields
			priority=EventPriority(event.priority),
			compression_type=CompressionType(event.compression_type),
			serialization_format=SerializationFormat(event.serialization_format),
			schema_id=event.schema_id,
			processing_duration_ms=event.processing_duration_ms,
			retry_count=event.retry_count,
			max_retries=event.max_retries,
			error_message=event.error_message
		)
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/events/batch", status_code=HTTP_201_CREATED)
async def publish_event_batch(
	request: EventBatchPublishRequest,
	background_tasks: BackgroundTasks,
	user: dict = Depends(get_current_user),
	publishing_service: EventPublishingService = Depends(get_event_publishing_service)
):
	"""Publish a batch of events for high-throughput scenarios."""
	try:
		# Convert requests to event configs
		event_configs = []
		for event_req in request.events:
			config = EventConfig(
				event_type=event_req.event_type,
				source_capability=event_req.source_capability,
				aggregate_id=event_req.aggregate_id,
				aggregate_type=event_req.aggregate_type,
				correlation_id=event_req.correlation_id,
				causation_id=event_req.causation_id,
				partition_key=event_req.partition_key,
				metadata=event_req.metadata
			)
			event_configs.append((config, event_req.payload))
		
		# Publish batch
		event_ids = await publishing_service.publish_event_batch(
			events=event_configs,
			stream_id=request.stream_id,
			tenant_id=user["tenant_id"],
			user_id=user["user_id"],
			batch_options=request.batch_options
		)
		
		# Background task to notify WebSocket subscribers
		background_tasks.add_task(
			notify_batch_published,
			event_ids,
			request.stream_id
		)
		
		return {
			"message": f"Successfully published {len(event_ids)} events",
			"event_ids": event_ids,
			"batch_size": len(event_ids)
		}
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

async def notify_batch_published(event_ids: List[str], stream_id: Optional[str]):
	"""Background task to notify WebSocket subscribers of batch publish."""
	if stream_id:
		message = WebSocketMessage(
			message_type="batch_published",
			data={
				"stream_id": stream_id,
				"event_count": len(event_ids),
				"event_ids": event_ids[:10]  # First 10 IDs only
			}
		)
		await connection_manager.broadcast_to_stream(stream_id, message)

# =============================================================================
# Event Query Endpoints
# =============================================================================

@app.get("/api/v1/events/{event_id}", response_model=EventResponse)
async def get_event(
	event_id: str,
	user: dict = Depends(get_current_user),
	publishing_service: EventPublishingService = Depends(get_event_publishing_service)
):
	"""Get a specific event by ID."""
	try:
		event = await publishing_service.get_event(event_id)
		if not event:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Event not found")
		
		return EventResponse(
			event_id=event.event_id,
			event_type=event.event_type,
			event_version=event.event_version,
			source_capability=event.source_capability,
			aggregate_id=event.aggregate_id,
			aggregate_type=event.aggregate_type,
			sequence_number=event.sequence_number,
			timestamp=event.timestamp,
			correlation_id=event.correlation_id,
			causation_id=event.causation_id,
			payload=event.payload,
			metadata=event.metadata,
			status=EventStatus(event.status),
			stream_id=event.stream_id,
			# Enhanced response fields
			priority=EventPriority(event.priority),
			compression_type=CompressionType(event.compression_type),
			serialization_format=SerializationFormat(event.serialization_format),
			schema_id=event.schema_id,
			processing_duration_ms=event.processing_duration_ms,
			retry_count=event.retry_count,
			max_retries=event.max_retries,
			error_message=event.error_message
		)
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/events/query")
async def query_events(
	query: EventQueryRequest,
	user: dict = Depends(get_current_user),
	streaming_service: EventStreamingService = Depends(get_event_streaming_service)
):
	"""Query events with advanced filtering options."""
	try:
		# Build filter criteria
		filters = {}
		if query.stream_id:
			filters["stream_id"] = query.stream_id
		if query.event_type:
			filters["event_type"] = query.event_type
		if query.source_capability:
			filters["source_capability"] = query.source_capability
		if query.aggregate_id:
			filters["aggregate_id"] = query.aggregate_id
		if query.aggregate_type:
			filters["aggregate_type"] = query.aggregate_type
		if query.correlation_id:
			filters["correlation_id"] = query.correlation_id
		if query.status:
			filters["status"] = query.status.value
		
		# Add tenant isolation
		filters["tenant_id"] = user["tenant_id"]
		
		# Query events
		events, total_count = await streaming_service.query_events(
			filters=filters,
			start_time=query.start_time,
			end_time=query.end_time,
			limit=query.limit,
			offset=query.offset
		)
		
		# Convert to response format
		event_responses = []
		for event in events:
			event_responses.append(EventResponse(
				event_id=event.event_id,
				event_type=event.event_type,
				event_version=event.event_version,
				source_capability=event.source_capability,
				aggregate_id=event.aggregate_id,
				aggregate_type=event.aggregate_type,
				sequence_number=event.sequence_number,
				timestamp=event.timestamp,
				correlation_id=event.correlation_id,
				causation_id=event.causation_id,
				payload=event.payload,
				metadata=event.metadata,
				status=EventStatus(event.status),
				stream_id=event.stream_id,
				# Enhanced response fields
				priority=EventPriority(event.priority),
				compression_type=CompressionType(event.compression_type),
				serialization_format=SerializationFormat(event.serialization_format),
				schema_id=event.schema_id,
				processing_duration_ms=event.processing_duration_ms,
				retry_count=event.retry_count,
				max_retries=event.max_retries,
				error_message=event.error_message
			))
		
		return {
			"events": event_responses,
			"total_count": total_count,
			"limit": query.limit,
			"offset": query.offset,
			"has_more": (query.offset + query.limit) < total_count
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Stream Management Endpoints
# =============================================================================

@app.get("/api/v1/streams")
async def list_streams(
	user: dict = Depends(get_current_user),
	streaming_service: EventStreamingService = Depends(get_event_streaming_service)
):
	"""List all available event streams."""
	try:
		streams = await streaming_service.list_streams(tenant_id=user["tenant_id"])
		return {"streams": streams}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/streams", status_code=HTTP_201_CREATED)
async def create_stream(
	config: StreamConfig,
	user: dict = Depends(get_current_user),
	streaming_service: EventStreamingService = Depends(get_event_streaming_service)
):
	"""Create a new event stream."""
	try:
		stream_id = await streaming_service.create_stream(
			config=config,
			tenant_id=user["tenant_id"],
			created_by=user["user_id"]
		)
		
		return {
			"stream_id": stream_id,
			"message": "Stream created successfully"
		}
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/streams/{stream_id}")
async def get_stream(
	stream_id: str,
	user: dict = Depends(get_current_user),
	streaming_service: EventStreamingService = Depends(get_event_streaming_service)
):
	"""Get stream configuration and details."""
	try:
		stream = await streaming_service.get_stream(stream_id, user["tenant_id"])
		if not stream:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Stream not found")
		
		return stream
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/streams/{stream_id}/events")
async def get_stream_events(
	stream_id: str,
	limit: int = Query(default=100, ge=1, le=10000),
	offset: int = Query(default=0, ge=0),
	start_time: Optional[datetime] = Query(None),
	end_time: Optional[datetime] = Query(None),
	user: dict = Depends(get_current_user),
	streaming_service: EventStreamingService = Depends(get_event_streaming_service)
):
	"""Get events from a specific stream."""
	try:
		events, total_count = await streaming_service.get_stream_events(
			stream_id=stream_id,
			tenant_id=user["tenant_id"],
			start_time=start_time,
			end_time=end_time,
			limit=limit,
			offset=offset
		)
		
		return {
			"events": events,
			"total_count": total_count,
			"stream_id": stream_id,
			"limit": limit,
			"offset": offset
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/streams/{stream_id}/metrics", response_model=StreamMetricsResponse)
async def get_stream_metrics(
	stream_id: str,
	user: dict = Depends(get_current_user),
	streaming_service: EventStreamingService = Depends(get_event_streaming_service)
):
	"""Get real-time metrics for a stream."""
	try:
		metrics = await streaming_service.get_stream_metrics(stream_id, user["tenant_id"])
		if not metrics:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Stream not found")
		
		return StreamMetricsResponse(**metrics)
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Subscription Management Endpoints
# =============================================================================

@app.post("/api/v1/subscriptions", status_code=HTTP_201_CREATED)
async def create_subscription(
	config: SubscriptionConfig,
	user: dict = Depends(get_current_user),
	consumption_service: EventConsumptionService = Depends(get_event_consumption_service)
):
	"""Create a new event subscription."""
	try:
		subscription_id = await consumption_service.create_subscription(
			config=config,
			tenant_id=user["tenant_id"],
			created_by=user["user_id"]
		)
		
		return {
			"subscription_id": subscription_id,
			"message": "Subscription created successfully"
		}
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/subscriptions")
async def list_subscriptions(
	user: dict = Depends(get_current_user),
	consumption_service: EventConsumptionService = Depends(get_event_consumption_service)
):
	"""List all subscriptions for the tenant."""
	try:
		subscriptions = await consumption_service.list_subscriptions(tenant_id=user["tenant_id"])
		return {"subscriptions": subscriptions}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/subscriptions/{subscription_id}/status", response_model=SubscriptionStatusResponse)
async def get_subscription_status(
	subscription_id: str,
	user: dict = Depends(get_current_user),
	consumption_service: EventConsumptionService = Depends(get_event_consumption_service)
):
	"""Get detailed status for a subscription."""
	try:
		status = await consumption_service.get_subscription_status(subscription_id, user["tenant_id"])
		if not status:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Subscription not found")
		
		return SubscriptionStatusResponse(**status)
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/subscriptions/{subscription_id}", status_code=HTTP_204_NO_CONTENT)
async def cancel_subscription(
	subscription_id: str,
	user: dict = Depends(get_current_user),
	consumption_service: EventConsumptionService = Depends(get_event_consumption_service)
):
	"""Cancel an active subscription."""
	try:
		success = await consumption_service.cancel_subscription(subscription_id, user["tenant_id"])
		if not success:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Subscription not found")
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Schema Registry Endpoints
# =============================================================================

@app.post("/api/v1/schemas", status_code=HTTP_201_CREATED)
async def register_schema(
	config: SchemaConfig,
	user: dict = Depends(get_current_user),
	schema_service: SchemaRegistryService = Depends(get_schema_registry_service)
):
	"""Register a new event schema."""
	try:
		schema_id = await schema_service.register_schema(
			config=config,
			tenant_id=user["tenant_id"],
			created_by=user["user_id"]
		)
		
		return {
			"schema_id": schema_id,
			"message": "Schema registered successfully"
		}
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/schemas")
async def list_schemas(
	event_type: Optional[str] = Query(None),
	user: dict = Depends(get_current_user),
	schema_service: SchemaRegistryService = Depends(get_schema_registry_service)
):
	"""List registered schemas."""
	try:
		schemas = await schema_service.list_schemas(
			tenant_id=user["tenant_id"],
			event_type=event_type
		)
		return {"schemas": schemas}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/schemas/{schema_id}")
async def get_schema(
	schema_id: str,
	user: dict = Depends(get_current_user),
	schema_service: SchemaRegistryService = Depends(get_schema_registry_service)
):
	"""Get schema definition by ID."""
	try:
		schema = await schema_service.get_schema(schema_id, user["tenant_id"])
		if not schema:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Schema not found")
		
		return schema
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/events/{stream_name}")
async def websocket_stream_events(websocket: WebSocket, stream_name: str):
	"""WebSocket endpoint for real-time event streaming."""
	connection_id = str(uuid4())
	
	try:
		await connection_manager.connect(websocket, connection_id)
		
		# Subscribe to stream events
		connection_manager.subscribe_to_stream(connection_id, stream_name)
		
		# Send connection confirmation
		welcome_message = WebSocketMessage(
			message_type="connected",
			data={"stream_name": stream_name, "connection_id": connection_id}
		)
		await websocket.send_text(welcome_message.model_dump_json())
		
		# Keep connection alive and handle incoming messages
		while True:
			try:
				data = await websocket.receive_text()
				message = json.loads(data)
				
				# Handle ping messages
				if message.get("type") == "ping":
					pong_message = WebSocketMessage(
						message_type="pong",
						data={"timestamp": datetime.utcnow().isoformat()}
					)
					await websocket.send_text(pong_message.model_dump_json())
				
			except WebSocketDisconnect:
				break
			except Exception as e:
				error_message = WebSocketMessage(
					message_type="error",
					data={"error": str(e)}
				)
				await websocket.send_text(error_message.model_dump_json())
	
	except WebSocketDisconnect:
		pass
	finally:
		connection_manager.disconnect(connection_id)

@app.websocket("/ws/subscriptions/{subscription_id}")
async def websocket_subscription_updates(websocket: WebSocket, subscription_id: str):
	"""WebSocket endpoint for subscription status updates."""
	connection_id = str(uuid4())
	
	try:
		await connection_manager.connect(websocket, connection_id)
		
		# Subscribe to subscription updates
		connection_manager.subscribe_to_subscription(connection_id, subscription_id)
		
		# Send connection confirmation
		welcome_message = WebSocketMessage(
			message_type="connected",
			data={"subscription_id": subscription_id, "connection_id": connection_id}
		)
		await websocket.send_text(welcome_message.model_dump_json())
		
		# Keep connection alive
		while True:
			try:
				data = await websocket.receive_text()
				message = json.loads(data)
				
				# Handle ping messages
				if message.get("type") == "ping":
					pong_message = WebSocketMessage(
						message_type="pong",
						data={"timestamp": datetime.utcnow().isoformat()}
					)
					await websocket.send_text(pong_message.model_dump_json())
				
			except WebSocketDisconnect:
				break
			except Exception as e:
				error_message = WebSocketMessage(
					message_type="error",
					data={"error": str(e)}
				)
				await websocket.send_text(error_message.model_dump_json())
	
	except WebSocketDisconnect:
		pass
	finally:
		connection_manager.disconnect(connection_id)

@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket):
	"""WebSocket endpoint for real-time monitoring data."""
	connection_id = str(uuid4())
	
	try:
		await connection_manager.connect(websocket, connection_id)
		
		# Send initial monitoring data
		while True:
			# Get current metrics
			monitoring_data = {
				"timestamp": datetime.utcnow().isoformat(),
				"active_connections": len(connection_manager.active_connections),
				"stream_subscribers": len(connection_manager.stream_subscribers),
				"subscription_subscribers": len(connection_manager.subscription_subscribers)
			}
			
			message = WebSocketMessage(
				message_type="monitoring",
				data=monitoring_data
			)
			
			await websocket.send_text(message.model_dump_json())
			
			# Wait 5 seconds before next update
			await asyncio.sleep(5)
	
	except WebSocketDisconnect:
		pass
	finally:
		connection_manager.disconnect(connection_id)

# =============================================================================
# Event Sourcing and CQRS Endpoints
# =============================================================================

class EventSourcingRequest(BaseModel):
	"""Request model for event sourcing operations."""
	aggregate_id: str = Field(..., min_length=1, max_length=100)
	aggregate_type: str = Field(..., min_length=1, max_length=100)
	expected_version: Optional[int] = Field(None, ge=0)
	event_data: Dict[str, Any] = Field(...)
	event_type: str = Field(..., min_length=1, max_length=100)
	correlation_id: Optional[str] = Field(None, max_length=100)
	metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class AggregateReconstructionRequest(BaseModel):
	"""Request model for aggregate reconstruction."""
	aggregate_id: str = Field(..., min_length=1, max_length=100)
	aggregate_type: str = Field(..., min_length=1, max_length=100)
	update_snapshots: bool = Field(default=False)

@app.post("/api/v1/event-sourcing/append", status_code=HTTP_201_CREATED)
async def append_event(
	request: EventSourcingRequest,
	user: dict = Depends(get_current_user),
	sourcing_service: EventSourcingService = Depends(get_event_sourcing_service)
):
	"""Append event to aggregate stream with optimistic concurrency control."""
	try:
		event_id = await sourcing_service.append_event(
			aggregate_id=request.aggregate_id,
			aggregate_type=request.aggregate_type,
			event_data=request.event_data,
			event_type=request.event_type,
			expected_version=request.expected_version,
			correlation_id=request.correlation_id,
			metadata=request.metadata,
			tenant_id=user["tenant_id"],
			user_id=user["user_id"]
		)
		
		return {
			"event_id": event_id,
			"message": "Event appended successfully"
		}
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/event-sourcing/reconstruct")
async def reconstruct_aggregate(
	request: AggregateReconstructionRequest,
	user: dict = Depends(get_current_user),
	sourcing_service: EventSourcingService = Depends(get_event_sourcing_service)
):
	"""Reconstruct aggregate state from event stream."""
	try:
		aggregate_state = await sourcing_service.reconstruct_aggregate(
			aggregate_id=request.aggregate_id,
			aggregate_type=request.aggregate_type,
			update_snapshots=request.update_snapshots,
			tenant_id=user["tenant_id"]
		)
		
		return {
			"aggregate_id": request.aggregate_id,
			"aggregate_type": request.aggregate_type,
			"current_version": aggregate_state.get("version", 0),
			"state": aggregate_state.get("data", {}),
			"last_event_timestamp": aggregate_state.get("last_event_timestamp")
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/event-sourcing/aggregate/{aggregate_id}/events")
async def get_aggregate_events(
	aggregate_id: str,
	aggregate_type: str = Query(...),
	start_version: Optional[int] = Query(None, ge=0),
	end_version: Optional[int] = Query(None, ge=0),
	user: dict = Depends(get_current_user),
	sourcing_service: EventSourcingService = Depends(get_event_sourcing_service)
):
	"""Get events for a specific aggregate."""
	try:
		events = await sourcing_service.get_aggregate_events(
			aggregate_id=aggregate_id,
			aggregate_type=aggregate_type,
			start_version=start_version,
			end_version=end_version,
			tenant_id=user["tenant_id"]
		)
		
		return {
			"aggregate_id": aggregate_id,
			"aggregate_type": aggregate_type,
			"events": events,
			"event_count": len(events)
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Stream Processing Endpoints
# =============================================================================

class StreamProcessorRequest(BaseModel):
	"""Request model for stream processor creation."""
	processor_name: str = Field(..., min_length=1, max_length=100)
	processor_description: Optional[str] = Field(None, max_length=500)
	processor_type: ProcessorType = Field(...)
	source_stream_id: str = Field(..., min_length=1, max_length=100)
	target_stream_id: Optional[str] = Field(None, max_length=100)
	processing_logic: Dict[str, Any] = Field(...)
	state_store_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
	window_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
	parallelism: int = Field(default=1, ge=1, le=100)
	error_handling_strategy: str = Field(default="retry")

@app.post("/api/v1/stream-processors", status_code=HTTP_201_CREATED)
async def create_stream_processor(
	request: StreamProcessorRequest,
	user: dict = Depends(get_current_user),
	stream_service: StreamManagementService = Depends(get_stream_management_service)
):
	"""Create a new stream processor."""
	try:
		processor_id = await stream_service.create_stream_processor(
			processor_config=request.dict(),
			tenant_id=user["tenant_id"],
			user_id=user["user_id"]
		)
		
		return {
			"processor_id": processor_id,
			"message": "Stream processor created successfully"
		}
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/stream-processors")
async def list_stream_processors(
	user: dict = Depends(get_current_user),
	stream_service: StreamManagementService = Depends(get_stream_management_service)
):
	"""List all stream processors for the tenant."""
	try:
		processors = await stream_service.list_stream_processors(tenant_id=user["tenant_id"])
		return {"processors": processors}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/stream-processors/{processor_id}/start")
async def start_stream_processor(
	processor_id: str,
	user: dict = Depends(get_current_user),
	stream_service: StreamManagementService = Depends(get_stream_management_service)
):
	"""Start a stream processor."""
	try:
		success = await stream_service.start_stream_processor(
			processor_id=processor_id,
			tenant_id=user["tenant_id"]
		)
		
		if not success:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Processor not found")
		
		return {"message": "Stream processor started successfully"}
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/stream-processors/{processor_id}/stop")
async def stop_stream_processor(
	processor_id: str,
	user: dict = Depends(get_current_user),
	stream_service: StreamManagementService = Depends(get_stream_management_service)
):
	"""Stop a stream processor."""
	try:
		success = await stream_service.stop_stream_processor(
			processor_id=processor_id,
			tenant_id=user["tenant_id"]
		)
		
		if not success:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Processor not found")
		
		return {"message": "Stream processor stopped successfully"}
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stream-processors/{processor_id}/metrics")
async def get_processor_metrics(
	processor_id: str,
	user: dict = Depends(get_current_user),
	stream_service: StreamManagementService = Depends(get_stream_management_service)
):
	"""Get metrics for a specific stream processor."""
	try:
		metrics = await stream_service.get_processor_metrics(
			processor_id=processor_id,
			tenant_id=user["tenant_id"]
		)
		
		if not metrics:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Processor not found")
		
		return metrics
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Enhanced Schema Registry Endpoints
# =============================================================================

class EnhancedSchemaRequest(BaseModel):
	"""Request model for enhanced schema registration."""
	schema_name: str = Field(..., min_length=1, max_length=100)
	schema_version: str = Field(..., min_length=1, max_length=20)
	json_schema: Dict[str, Any] = Field(...)
	event_type: str = Field(..., min_length=1, max_length=100)
	compatibility_level: str = Field(default="BACKWARD")
	evolution_strategy: str = Field(default="COMPATIBLE")
	validation_rules: Optional[Dict[str, Any]] = Field(default_factory=dict)
	is_active: bool = Field(default=True)

@app.post("/api/v1/schemas/enhanced", status_code=HTTP_201_CREATED)
async def register_enhanced_schema(
	request: EnhancedSchemaRequest,
	user: dict = Depends(get_current_user),
	schema_service: SchemaRegistryService = Depends(get_schema_registry_service)
):
	"""Register an enhanced event schema with evolution support."""
	try:
		schema_id = await schema_service.register_enhanced_schema(
			schema_config=request.dict(),
			tenant_id=user["tenant_id"],
			created_by=user["user_id"]
		)
		
		return {
			"schema_id": schema_id,
			"message": "Enhanced schema registered successfully"
		}
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/schemas/{schema_id}/validate")
async def validate_event_against_schema(
	schema_id: str,
	event_data: Dict[str, Any],
	user: dict = Depends(get_current_user),
	schema_service: SchemaRegistryService = Depends(get_schema_registry_service)
):
	"""Validate event data against a specific schema."""
	try:
		validation_result = await schema_service.validate_event(
			schema_id=schema_id,
			event_data=event_data,
			tenant_id=user["tenant_id"]
		)
		
		return validation_result
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/schemas/evolution/{event_type}")
async def get_schema_evolution(
	event_type: str,
	user: dict = Depends(get_current_user),
	schema_service: SchemaRegistryService = Depends(get_schema_registry_service)
):
	"""Get schema evolution history for an event type."""
	try:
		evolution = await schema_service.get_schema_evolution(
			event_type=event_type,
			tenant_id=user["tenant_id"]
		)
		
		return {
			"event_type": event_type,
			"evolution_history": evolution
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Consumer Group Management Endpoints
# =============================================================================

class ConsumerGroupRequest(BaseModel):
	"""Request model for consumer group management."""
	group_name: str = Field(..., min_length=1, max_length=100)
	group_description: Optional[str] = Field(None, max_length=500)
	session_timeout_ms: int = Field(default=30000, ge=1000, le=300000)
	heartbeat_interval_ms: int = Field(default=3000, ge=100, le=30000)
	max_poll_interval_ms: int = Field(default=300000, ge=10000, le=1800000)
	partition_assignment_strategy: str = Field(default="range")
	rebalance_timeout_ms: int = Field(default=60000, ge=1000, le=600000)

@app.post("/api/v1/consumer-groups", status_code=HTTP_201_CREATED)
async def create_consumer_group(
	request: ConsumerGroupRequest,
	user: dict = Depends(get_current_user),
	consumer_service: ConsumerManagementService = Depends(get_consumer_management_service)
):
	"""Create a new consumer group."""
	try:
		group_id = await consumer_service.create_consumer_group(
			group_config=request.dict(),
			tenant_id=user["tenant_id"],
			user_id=user["user_id"]
		)
		
		return {
			"group_id": group_id,
			"message": "Consumer group created successfully"
		}
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/consumer-groups")
async def list_consumer_groups(
	user: dict = Depends(get_current_user),
	consumer_service: ConsumerManagementService = Depends(get_consumer_management_service)
):
	"""List all consumer groups for the tenant."""
	try:
		groups = await consumer_service.list_consumer_groups(tenant_id=user["tenant_id"])
		return {"consumer_groups": groups}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/consumer-groups/{group_id}/lag")
async def get_consumer_group_lag(
	group_id: str,
	user: dict = Depends(get_current_user),
	consumer_service: ConsumerManagementService = Depends(get_consumer_management_service)
):
	"""Get consumer lag for a specific group."""
	try:
		lag_info = await consumer_service.get_consumer_lag(
			group_id=group_id,
			tenant_id=user["tenant_id"]
		)
		
		if not lag_info:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Consumer group not found")
		
		return lag_info
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/consumer-groups/{group_id}/rebalance")
async def trigger_consumer_rebalance(
	group_id: str,
	user: dict = Depends(get_current_user),
	consumer_service: ConsumerManagementService = Depends(get_consumer_management_service)
):
	"""Trigger a rebalance for a consumer group."""
	try:
		success = await consumer_service.trigger_rebalance(
			group_id=group_id,
			tenant_id=user["tenant_id"]
		)
		
		if not success:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Consumer group not found")
		
		return {"message": "Consumer group rebalance triggered successfully"}
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Health Check and Status Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
	"""Health check endpoint."""
	return {
		"status": "healthy",
		"timestamp": datetime.utcnow().isoformat(),
		"version": "1.0.0"
	}

@app.get("/api/v1/status")
async def get_system_status():
	"""Get overall system status and metrics."""
	try:
		# In production, this would check various system components
		status = {
			"system_status": "operational",
			"timestamp": datetime.utcnow().isoformat(),
			"components": {
				"api": "healthy",
				"kafka": "healthy",
				"redis": "healthy",
				"postgresql": "healthy"
			},
			"active_connections": len(connection_manager.active_connections),
			"stream_subscribers": len(connection_manager.stream_subscribers),
			"subscription_subscribers": len(connection_manager.subscription_subscribers)
		}
		
		return status
		
	except Exception as e:
		return {
			"system_status": "degraded",
			"timestamp": datetime.utcnow().isoformat(),
			"error": str(e)
		}

# Create router for integration with other frameworks
from fastapi import APIRouter
router = APIRouter()

# Add all routes to router
for route in app.routes:
	if hasattr(route, 'path'):
		router.routes.append(route)

# Export main components
__all__ = [
	'app',
	'router',
	'ConnectionManager',
	'connection_manager',
	'EventPublishRequest',
	'EventBatchPublishRequest',
	'EventQueryRequest',
	'EventResponse',
	'WebSocketMessage',
	'EventSourcingRequest',
	'AggregateReconstructionRequest',
	'StreamProcessorRequest',
	'EnhancedSchemaRequest',
	'ConsumerGroupRequest'
]
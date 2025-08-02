"""
APG Central Configuration - Real-Time Configuration Synchronization

Advanced real-time synchronization system with WebSocket support, operational transforms,
conflict resolution, and multi-region consistency.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str
import logging
from contextlib import asynccontextmanager

# WebSocket and real-time support
import websockets
from websockets.server import serve
from websockets.exceptions import ConnectionClosed, WebSocketException

# Message queuing and event streaming
import redis.asyncio as redis
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import asyncio_mqtt

# APG integrations
from ..common.real_time_collaboration.websocket_manager import websocket_manager, MessageType
from .error_handling import ErrorHandler, ErrorCategory, ErrorSeverity, with_error_handling

logger = logging.getLogger(__name__)

class SyncEventType(Enum):
	"""Types of synchronization events."""
	CONFIG_CHANGED = "config_changed"
	CONFIG_CREATED = "config_created"
	CONFIG_DELETED = "config_deleted"
	CONFIG_LOCKED = "config_locked"
	CONFIG_UNLOCKED = "config_unlocked"
	BATCH_UPDATE = "batch_update"
	SCHEMA_UPDATED = "schema_updated"
	ROLLBACK = "rollback"
	CONFLICT_DETECTED = "conflict_detected"

class ConflictResolutionStrategy(Enum):
	"""Conflict resolution strategies."""
	LAST_WRITE_WINS = "last_write_wins"
	FIRST_WRITE_WINS = "first_write_wins"
	MANUAL_RESOLUTION = "manual_resolution"
	OPERATIONAL_TRANSFORM = "operational_transform"
	MERGE_STRATEGIES = "merge_strategies"

class SyncScope(Enum):
	"""Synchronization scope levels."""
	GLOBAL = "global"
	TENANT = "tenant"
	USER = "user"
	SESSION = "session"

@dataclass
class SyncEvent:
	"""Real-time synchronization event."""
	event_id: str = field(default_factory=uuid7str)
	event_type: SyncEventType = SyncEventType.CONFIG_CHANGED
	timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	source_node: str = "unknown"
	tenant_id: Optional[str] = None
	user_id: Optional[str] = None
	session_id: Optional[str] = None
	config_key: Optional[str] = None
	old_value: Any = None
	new_value: Any = None
	version: int = 1
	checksum: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
	requires_ack: bool = True
	retry_count: int = 0
	max_retries: int = 3

@dataclass
class OperationalTransform:
	"""Operational transform for conflict resolution."""
	operation_id: str = field(default_factory=uuid7str)
	operation_type: str = "update"  # update, insert, delete, move
	path: str = ""  # JSON path to the affected data
	old_value: Any = None
	new_value: Any = None
	position: int = 0
	priority: int = 0
	timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	author: str = "system"

@dataclass
class ConflictInfo:
	"""Information about configuration conflicts."""
	conflict_id: str = field(default_factory=uuid7str)
	config_key: str = ""
	competing_versions: List[Dict[str, Any]] = field(default_factory=list)
	resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS
	resolved: bool = False
	resolution_result: Any = None
	created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	resolved_at: Optional[datetime] = None
	resolved_by: Optional[str] = None

class RealtimeSyncManager:
	"""Comprehensive real-time configuration synchronization manager."""
	
	def __init__(
		self,
		redis_client: redis.Redis,
		kafka_bootstrap_servers: Optional[List[str]] = None,
		mqtt_broker_host: Optional[str] = None,
		mqtt_broker_port: int = 1883,
		node_id: Optional[str] = None
	):
		self.redis_client = redis_client
		self.kafka_bootstrap_servers = kafka_bootstrap_servers or ["localhost:9092"]
		self.mqtt_broker_host = mqtt_broker_host or "localhost"
		self.mqtt_broker_port = mqtt_broker_port
		self.node_id = node_id or f"node_{uuid7str()[:8]}"
		
		# Error handling
		self.error_handler = ErrorHandler(f"realtime_sync_{self.node_id}")
		
		# Active connections and subscriptions
		self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
		self.subscription_patterns: Dict[str, Set[str]] = {}  # connection_id -> patterns
		self.active_locks: Dict[str, Dict[str, Any]] = {}  # config_key -> lock_info
		self.conflict_handlers: Dict[str, Callable] = {}
		
		# Operational transforms buffer
		self.pending_transforms: Dict[str, List[OperationalTransform]] = {}
		self.applied_transforms: Dict[str, List[OperationalTransform]] = {}
		
		# Synchronization state
		self.last_sync_timestamps: Dict[str, datetime] = {}
		self.sync_checkpoints: Dict[str, str] = {}
		
		# Event handlers
		self.event_handlers: Dict[SyncEventType, List[Callable]] = {}
		
		# Message queuing
		self.kafka_producer: Optional[AIOKafkaProducer] = None
		self.kafka_consumer: Optional[AIOKafkaConsumer] = None
		self.mqtt_client: Optional[asyncio_mqtt.Client] = None
		
		# Initialize event handlers
		self._setup_default_event_handlers()
	
	def _setup_default_event_handlers(self):
		"""Setup default event handlers."""
		self.event_handlers = {
			SyncEventType.CONFIG_CHANGED: [self._handle_config_changed],
			SyncEventType.CONFIG_CREATED: [self._handle_config_created],
			SyncEventType.CONFIG_DELETED: [self._handle_config_deleted],
			SyncEventType.CONFIG_LOCKED: [self._handle_config_locked],
			SyncEventType.CONFIG_UNLOCKED: [self._handle_config_unlocked],
			SyncEventType.CONFLICT_DETECTED: [self._handle_conflict_detected],
			SyncEventType.BATCH_UPDATE: [self._handle_batch_update],
		}
	
	@with_error_handling(ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH)
	async def initialize(self):
		"""Initialize all synchronization components."""
		try:
			# Initialize Kafka producer/consumer
			if self.kafka_bootstrap_servers:
				await self._initialize_kafka()
			
			# Initialize MQTT client
			if self.mqtt_broker_host:
				await self._initialize_mqtt()
			
			# Start background tasks
			asyncio.create_task(self._sync_heartbeat_task())
			asyncio.create_task(self._conflict_resolution_task())
			asyncio.create_task(self._cleanup_task())
			
			logger.info(f"Real-time sync manager initialized for node {self.node_id}")
			
		except Exception as e:
			await self.error_handler.handle_error(
				e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.CRITICAL,
				"initialize_realtime_sync", {"node_id": self.node_id}
			)
			raise
	
	async def _initialize_kafka(self):
		"""Initialize Kafka producer and consumer."""
		try:
			self.kafka_producer = AIOKafkaProducer(
				bootstrap_servers=self.kafka_bootstrap_servers,
				value_serializer=lambda x: json.dumps(x).encode('utf-8'),
				key_serializer=lambda x: x.encode('utf-8') if x else None,
				compression_type="gzip",
				max_request_size=10485760,  # 10MB
				request_timeout_ms=30000,
				retry_backoff_ms=1000,
				retries=3
			)
			await self.kafka_producer.start()
			
			self.kafka_consumer = AIOKafkaConsumer(
				"apg_config_sync",
				"apg_config_conflicts",
				bootstrap_servers=self.kafka_bootstrap_servers,
				group_id=f"sync_group_{self.node_id}",
				value_deserializer=lambda x: json.loads(x.decode('utf-8')),
				auto_offset_reset='latest',
				enable_auto_commit=True,
				auto_commit_interval_ms=1000
			)
			await self.kafka_consumer.start()
			
			# Start consumer task
			asyncio.create_task(self._kafka_message_handler())
			
		except Exception as e:
			await self.error_handler.handle_error(
				e, ErrorCategory.EXTERNAL_SERVICE_ERROR, ErrorSeverity.HIGH,
				"initialize_kafka", {"bootstrap_servers": self.kafka_bootstrap_servers}
			)
			raise
	
	async def _initialize_mqtt(self):
		"""Initialize MQTT client for lightweight messaging."""
		try:
			self.mqtt_client = asyncio_mqtt.Client(
				hostname=self.mqtt_broker_host,
				port=self.mqtt_broker_port,
				client_id=f"apg_sync_{self.node_id}",
				keepalive=60,
				will=asyncio_mqtt.Will(
					topic=f"apg/config/sync/{self.node_id}/status",
					payload="offline",
					qos=1,
					retain=True
				)
			)
			
			# Start MQTT task
			asyncio.create_task(self._mqtt_message_handler())
			
		except Exception as e:
			await self.error_handler.handle_error(
				e, ErrorCategory.EXTERNAL_SERVICE_ERROR, ErrorSeverity.MEDIUM,
				"initialize_mqtt", {"broker": f"{self.mqtt_broker_host}:{self.mqtt_broker_port}"}
			)
			raise
	
	@with_error_handling(ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM)
	async def add_websocket_connection(
		self,
		connection_id: str,
		websocket: websockets.WebSocketServerProtocol,
		subscription_patterns: Optional[List[str]] = None
	):
		"""Add WebSocket connection for real-time updates."""
		self.websocket_connections[connection_id] = websocket
		self.subscription_patterns[connection_id] = set(subscription_patterns or ["*"])
		
		logger.info(f"Added WebSocket connection {connection_id} with patterns: {subscription_patterns}")
		
		# Send initial sync data
		await self._send_initial_sync_data(connection_id)
	
	async def remove_websocket_connection(self, connection_id: str):
		"""Remove WebSocket connection."""
		if connection_id in self.websocket_connections:
			del self.websocket_connections[connection_id]
		if connection_id in self.subscription_patterns:
			del self.subscription_patterns[connection_id]
		
		logger.info(f"Removed WebSocket connection {connection_id}")
	
	@with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.MEDIUM)
	async def broadcast_sync_event(self, event: SyncEvent):
		"""Broadcast synchronization event to all relevant connections."""
		try:
			# Publish to Redis for local broadcasting
			await self.redis_client.publish(
				f"apg:config:sync:{event.tenant_id or 'global'}",
				json.dumps({
					"event_id": event.event_id,
					"event_type": event.event_type.value,
					"timestamp": event.timestamp.isoformat(),
					"source_node": event.source_node,
					"config_key": event.config_key,
					"old_value": event.old_value,
					"new_value": event.new_value,
					"version": event.version,
					"metadata": event.metadata
				})
			)
			
			# Publish to Kafka for cross-region sync
			if self.kafka_producer:
				await self.kafka_producer.send(
					"apg_config_sync",
					key=event.config_key or event.event_id,
					value={
						"event_id": event.event_id,
						"event_type": event.event_type.value,
						"timestamp": event.timestamp.isoformat(),
						"source_node": event.source_node,
						"tenant_id": event.tenant_id,
						"config_key": event.config_key,
						"old_value": event.old_value,
						"new_value": event.new_value,
						"version": event.version,
						"checksum": event.checksum,
						"metadata": event.metadata
					}
				)
			
			# Send to MQTT for lightweight clients
			if self.mqtt_client:
				topic = f"apg/config/{event.tenant_id or 'global'}/{event.config_key or 'system'}"
				await self.mqtt_client.publish(
					topic,
					payload=json.dumps({
						"event_type": event.event_type.value,
						"config_key": event.config_key,
						"new_value": event.new_value,
						"version": event.version,
						"timestamp": event.timestamp.isoformat()
					}),
					qos=1
				)
			
			# Broadcast to WebSocket connections
			await self._broadcast_to_websockets(event)
			
			# Trigger event handlers
			handlers = self.event_handlers.get(event.event_type, [])
			for handler in handlers:
				try:
					await handler(event)
				except Exception as e:
					await self.error_handler.handle_error(
						e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM,
						f"event_handler_{handler.__name__}",
						{"event_id": event.event_id, "event_type": event.event_type.value}
					)
			
		except Exception as e:
			await self.error_handler.handle_error(
				e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH,
				"broadcast_sync_event",
				{"event_id": event.event_id, "event_type": event.event_type.value}
			)
			raise
	
	async def _broadcast_to_websockets(self, event: SyncEvent):
		"""Broadcast event to matching WebSocket connections."""
		dead_connections = []
		
		for connection_id, websocket in self.websocket_connections.items():
			try:
				# Check if connection matches subscription patterns
				patterns = self.subscription_patterns.get(connection_id, set())
				if self._matches_patterns(event.config_key or "", patterns):
					message = {
						"type": "config_sync",
						"event_type": event.event_type.value,
						"event_id": event.event_id,
						"timestamp": event.timestamp.isoformat(),
						"config_key": event.config_key,
						"old_value": event.old_value,
						"new_value": event.new_value,
						"version": event.version,
						"metadata": event.metadata
					}
					
					await websocket.send(json.dumps(message))
					
			except (ConnectionClosed, WebSocketException):
				dead_connections.append(connection_id)
			except Exception as e:
				await self.error_handler.handle_error(
					e, ErrorCategory.NETWORK_ERROR, ErrorSeverity.LOW,
					"websocket_broadcast",
					{"connection_id": connection_id, "event_id": event.event_id}
				)
				dead_connections.append(connection_id)
		
		# Clean up dead connections
		for connection_id in dead_connections:
			await self.remove_websocket_connection(connection_id)
	
	def _matches_patterns(self, config_key: str, patterns: Set[str]) -> bool:
		"""Check if config key matches any subscription patterns."""
		import fnmatch
		
		for pattern in patterns:
			if pattern == "*" or fnmatch.fnmatch(config_key, pattern):
				return True
		return False
	
	@with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.MEDIUM)
	async def acquire_config_lock(
		self,
		config_key: str,
		user_id: str,
		tenant_id: Optional[str] = None,
		timeout: int = 300
	) -> bool:
		"""Acquire exclusive lock on configuration for editing."""
		lock_key = f"lock:{config_key}"
		lock_info = {
			"user_id": user_id,
			"tenant_id": tenant_id,
			"acquired_at": datetime.now(timezone.utc).isoformat(),
			"expires_at": (datetime.now(timezone.utc) + timedelta(seconds=timeout)).isoformat(),
			"node_id": self.node_id
		}
		
		# Try to acquire lock in Redis
		acquired = await self.redis_client.set(
			lock_key,
			json.dumps(lock_info),
			nx=True,  # Only set if key doesn't exist
			ex=timeout  # Expire after timeout seconds
		)
		
		if acquired:
			self.active_locks[config_key] = lock_info
			
			# Broadcast lock acquisition
			event = SyncEvent(
				event_type=SyncEventType.CONFIG_LOCKED,
				source_node=self.node_id,
				tenant_id=tenant_id,
				user_id=user_id,
				config_key=config_key,
				metadata={"lock_info": lock_info}
			)
			await self.broadcast_sync_event(event)
			
			return True
		
		return False
	
	@with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.LOW)
	async def release_config_lock(
		self,
		config_key: str,
		user_id: str,
		tenant_id: Optional[str] = None
	) -> bool:
		"""Release configuration lock."""
		lock_key = f"lock:{config_key}"
		
		# Check if we own the lock
		current_lock = await self.redis_client.get(lock_key)
		if current_lock:
			lock_info = json.loads(current_lock)
			if lock_info.get("user_id") != user_id or lock_info.get("node_id") != self.node_id:
				return False  # Don't own the lock
		
		# Release lock
		released = await self.redis_client.delete(lock_key)
		
		if released and config_key in self.active_locks:
			del self.active_locks[config_key]
			
			# Broadcast lock release
			event = SyncEvent(
				event_type=SyncEventType.CONFIG_UNLOCKED,
				source_node=self.node_id,
				tenant_id=tenant_id,
				user_id=user_id,
				config_key=config_key
			)
			await self.broadcast_sync_event(event)
			
			return True
		
		return False
	
	@with_error_handling(ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.HIGH)
	async def detect_and_resolve_conflicts(
		self,
		config_key: str,
		competing_updates: List[Dict[str, Any]],
		resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS
	) -> ConflictInfo:
		"""Detect and resolve configuration conflicts."""
		conflict = ConflictInfo(
			config_key=config_key,
			competing_versions=competing_updates,
			resolution_strategy=resolution_strategy
		)
		
		try:
			if resolution_strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
				# Choose the version with the latest timestamp
				latest_update = max(
					competing_updates,
					key=lambda x: datetime.fromisoformat(x.get("timestamp", "1970-01-01T00:00:00Z"))
				)
				conflict.resolution_result = latest_update
				conflict.resolved = True
				conflict.resolved_at = datetime.now(timezone.utc)
				conflict.resolved_by = "system_last_write_wins"
				
			elif resolution_strategy == ConflictResolutionStrategy.OPERATIONAL_TRANSFORM:
				# Apply operational transforms to merge changes
				resolved_value = await self._apply_operational_transforms(
					config_key, competing_updates
				)
				conflict.resolution_result = resolved_value
				conflict.resolved = True
				conflict.resolved_at = datetime.now(timezone.utc)
				conflict.resolved_by = "system_operational_transform"
				
			elif resolution_strategy == ConflictResolutionStrategy.MERGE_STRATEGIES:
				# Attempt intelligent merging based on data types
				merged_value = await self._merge_conflicting_values(competing_updates)
				conflict.resolution_result = merged_value
				conflict.resolved = True
				conflict.resolved_at = datetime.now(timezone.utc)
				conflict.resolved_by = "system_merge_strategy"
			
			# Broadcast conflict resolution
			if conflict.resolved:
				event = SyncEvent(
					event_type=SyncEventType.CONFLICT_DETECTED,
					source_node=self.node_id,
					config_key=config_key,
					new_value=conflict.resolution_result,
					metadata={
						"conflict_id": conflict.conflict_id,
						"resolution_strategy": resolution_strategy.value,
						"competing_versions_count": len(competing_updates)
					}
				)
				await self.broadcast_sync_event(event)
			
			return conflict
			
		except Exception as e:
			await self.error_handler.handle_error(
				e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH,
				"resolve_config_conflict",
				{
					"config_key": config_key,
					"conflict_id": conflict.conflict_id,
					"strategy": resolution_strategy.value
				}
			)
			raise
	
	async def _apply_operational_transforms(
		self,
		config_key: str,
		competing_updates: List[Dict[str, Any]]
	) -> Any:
		"""Apply operational transforms to resolve conflicts."""
		# This is a simplified implementation
		# In production, you'd implement full operational transform algorithms
		
		base_value = competing_updates[0].get("old_value", {})
		transforms = []
		
		for update in competing_updates:
			transform = OperationalTransform(
				operation_type="update",
				path=config_key,
				old_value=update.get("old_value"),
				new_value=update.get("new_value"),
				timestamp=datetime.fromisoformat(update.get("timestamp", datetime.now(timezone.utc).isoformat())),
				author=update.get("user_id", "unknown")
			)
			transforms.append(transform)
		
		# Sort transforms by timestamp
		transforms.sort(key=lambda t: t.timestamp)
		
		# Apply transforms sequentially
		result = base_value
		for transform in transforms:
			if isinstance(result, dict) and isinstance(transform.new_value, dict):
				result.update(transform.new_value)
			else:
				result = transform.new_value
		
		return result
	
	async def _merge_conflicting_values(self, competing_updates: List[Dict[str, Any]]) -> Any:
		"""Merge conflicting values using intelligent strategies."""
		if not competing_updates:
			return None
		
		# Get all new values
		new_values = [update.get("new_value") for update in competing_updates]
		
		# If all values are dictionaries, merge them
		if all(isinstance(v, dict) for v in new_values):
			merged = {}
			for value in new_values:
				merged.update(value)
			return merged
		
		# If all values are lists, concatenate and deduplicate
		elif all(isinstance(v, list) for v in new_values):
			merged_list = []
			for value in new_values:
				merged_list.extend(value)
			return list(set(merged_list))  # Remove duplicates
		
		# For scalar values, use last write wins
		else:
			latest_update = max(
				competing_updates,
				key=lambda x: datetime.fromisoformat(x.get("timestamp", "1970-01-01T00:00:00Z"))
			)
			return latest_update.get("new_value")
	
	# Event handlers
	async def _handle_config_changed(self, event: SyncEvent):
		"""Handle configuration change events."""
		logger.debug(f"Config changed: {event.config_key} -> {event.new_value}")
	
	async def _handle_config_created(self, event: SyncEvent):
		"""Handle configuration creation events."""
		logger.debug(f"Config created: {event.config_key}")
	
	async def _handle_config_deleted(self, event: SyncEvent):
		"""Handle configuration deletion events."""
		logger.debug(f"Config deleted: {event.config_key}")
	
	async def _handle_config_locked(self, event: SyncEvent):
		"""Handle configuration lock events."""
		logger.debug(f"Config locked: {event.config_key} by {event.user_id}")
	
	async def _handle_config_unlocked(self, event: SyncEvent):
		"""Handle configuration unlock events."""
		logger.debug(f"Config unlocked: {event.config_key}")
	
	async def _handle_conflict_detected(self, event: SyncEvent):
		"""Handle conflict detection events."""
		logger.warning(f"Conflict detected for: {event.config_key}")
	
	async def _handle_batch_update(self, event: SyncEvent):
		"""Handle batch update events."""
		logger.debug(f"Batch update processed: {len(event.metadata.get('updates', []))} configs")
	
	# Background tasks
	async def _sync_heartbeat_task(self):
		"""Send periodic heartbeat for node health monitoring."""
		while True:
			try:
				await asyncio.sleep(30)  # 30 second heartbeat
				
				heartbeat = {
					"node_id": self.node_id,
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"active_connections": len(self.websocket_connections),
					"active_locks": len(self.active_locks),
					"status": "healthy"
				}
				
				# Publish heartbeat to Redis
				await self.redis_client.publish("apg:config:heartbeat", json.dumps(heartbeat))
				
			except Exception as e:
				await self.error_handler.handle_error(
					e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.LOW,
					"sync_heartbeat", {"node_id": self.node_id}
				)
	
	async def _conflict_resolution_task(self):
		"""Background task for automatic conflict resolution."""
		while True:
			try:
				await asyncio.sleep(10)  # Check every 10 seconds
				
				# Check for pending conflicts in Redis
				conflict_keys = await self.redis_client.keys("conflict:*")
				
				for key_bytes in conflict_keys:
					key = key_bytes.decode('utf-8')
					conflict_data = await self.redis_client.get(key)
					
					if conflict_data:
						conflict_info = json.loads(conflict_data)
						if not conflict_info.get("resolved", False):
							# Attempt automatic resolution
							await self._attempt_auto_resolution(conflict_info)
				
			except Exception as e:
				await self.error_handler.handle_error(
					e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.LOW,
					"conflict_resolution_task", {}
				)
	
	async def _cleanup_task(self):
		"""Background cleanup task for expired locks and old data."""
		while True:
			try:
				await asyncio.sleep(300)  # Clean up every 5 minutes
				
				# Clean up expired locks
				current_time = datetime.now(timezone.utc)
				expired_locks = []
				
				for config_key, lock_info in self.active_locks.items():
					expires_at = datetime.fromisoformat(lock_info["expires_at"])
					if current_time > expires_at:
						expired_locks.append(config_key)
				
				for config_key in expired_locks:
					await self.release_config_lock(
						config_key,
						self.active_locks[config_key]["user_id"],
						self.active_locks[config_key].get("tenant_id")
					)
				
				# Clean up old sync events and conflicts
				await self._cleanup_old_events()
				
			except Exception as e:
				await self.error_handler.handle_error(
					e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.LOW,
					"cleanup_task", {}
				)
	
	async def _cleanup_old_events(self):
		"""Clean up old synchronization events and conflicts."""
		cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
		
		# Clean up old conflict records
		conflict_keys = await self.redis_client.keys("conflict:*")
		for key_bytes in conflict_keys:
			key = key_bytes.decode('utf-8')
			conflict_data = await self.redis_client.get(key)
			
			if conflict_data:
				conflict_info = json.loads(conflict_data)
				created_at = datetime.fromisoformat(conflict_info.get("created_at", cutoff_time.isoformat()))
				
				if created_at < cutoff_time and conflict_info.get("resolved", False):
					await self.redis_client.delete(key)
	
	async def _attempt_auto_resolution(self, conflict_info: Dict[str, Any]):
		"""Attempt automatic conflict resolution."""
		# This would implement more sophisticated auto-resolution logic
		pass
	
	async def _send_initial_sync_data(self, connection_id: str):
		"""Send initial synchronization data to new connection."""
		# This would send current configuration state to new connections
		pass
	
	async def _kafka_message_handler(self):
		"""Handle incoming Kafka messages."""
		if not self.kafka_consumer:
			return
		
		try:
			async for message in self.kafka_consumer:
				try:
					event_data = message.value
					event = SyncEvent(
						event_id=event_data.get("event_id"),
						event_type=SyncEventType(event_data.get("event_type")),
						timestamp=datetime.fromisoformat(event_data.get("timestamp")),
						source_node=event_data.get("source_node"),
						tenant_id=event_data.get("tenant_id"),
						config_key=event_data.get("config_key"),
						old_value=event_data.get("old_value"),
						new_value=event_data.get("new_value"),
						version=event_data.get("version", 1),
						metadata=event_data.get("metadata", {})
					)
					
					# Skip events from our own node to avoid loops
					if event.source_node != self.node_id:
						await self._process_remote_sync_event(event)
				
				except Exception as e:
					await self.error_handler.handle_error(
						e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM,
						"kafka_message_processing", {"message": str(message)}
					)
		
		except Exception as e:
			await self.error_handler.handle_error(
				e, ErrorCategory.EXTERNAL_SERVICE_ERROR, ErrorSeverity.HIGH,
				"kafka_message_handler", {}
			)
	
	async def _mqtt_message_handler(self):
		"""Handle incoming MQTT messages."""
		if not self.mqtt_client:
			return
		
		try:
			async with self.mqtt_client:
				# Subscribe to configuration topics
				await self.mqtt_client.subscribe("apg/config/+/+")
				await self.mqtt_client.subscribe(f"apg/config/sync/+/status")
				
				async for message in self.mqtt_client.messages:
					try:
						payload = json.loads(message.payload.decode())
						# Process MQTT sync messages
						await self._process_mqtt_sync_message(message.topic, payload)
					
					except Exception as e:
						await self.error_handler.handle_error(
							e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.LOW,
							"mqtt_message_processing", {"topic": str(message.topic)}
						)
		
		except Exception as e:
			await self.error_handler.handle_error(
				e, ErrorCategory.EXTERNAL_SERVICE_ERROR, ErrorSeverity.MEDIUM,
				"mqtt_message_handler", {}
			)
	
	async def _process_remote_sync_event(self, event: SyncEvent):
		"""Process synchronization event from remote node."""
		# Apply the remote change locally if needed
		# This would integrate with the configuration service
		pass
	
	async def _process_mqtt_sync_message(self, topic: str, payload: Dict[str, Any]):
		"""Process MQTT synchronization message."""
		# Handle lightweight sync messages from MQTT
		pass
	
	async def close(self):
		"""Clean up resources."""
		# Close Kafka connections
		if self.kafka_producer:
			await self.kafka_producer.stop()
		if self.kafka_consumer:
			await self.kafka_consumer.stop()
		
		# Close MQTT connection
		if self.mqtt_client:
			await self.mqtt_client.disconnect()
		
		# Close WebSocket connections
		for connection in self.websocket_connections.values():
			await connection.close()
		
		logger.info(f"Real-time sync manager closed for node {self.node_id}")

# Factory functions
async def create_realtime_sync_manager(
	redis_client: redis.Redis,
	kafka_bootstrap_servers: Optional[List[str]] = None,
	mqtt_broker_host: Optional[str] = None,
	node_id: Optional[str] = None
) -> RealtimeSyncManager:
	"""Create and initialize real-time synchronization manager."""
	manager = RealtimeSyncManager(
		redis_client=redis_client,
		kafka_bootstrap_servers=kafka_bootstrap_servers,
		mqtt_broker_host=mqtt_broker_host,
		node_id=node_id
	)
	
	await manager.initialize()
	return manager
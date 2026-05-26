"""
WebSocket Real-time Communications for the AI Core Framework (AICR) Capability
==============================================================================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

WebSocket server providing real-time bidirectional communication for AI operations,
monitoring events, pipeline status updates, inference streaming, and collaborative
features with intelligent message routing and connection management.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Set, Callable
from uuid import UUID

try:
	import websockets
	from websockets.server import WebSocketServerProtocol
except ImportError:
	class WebSocketServerProtocol:  # type: ignore[no-redef]
		pass

	class _ConnectionClosed(Exception):
		pass

	class _CompatWebSocketServer:
		def close(self) -> None:
			return None

		async def wait_closed(self) -> None:
			return None

	class _CompatWebSockets:
		WebSocketServer = _CompatWebSocketServer

		class exceptions:
			ConnectionClosed = _ConnectionClosed

		async def serve(self, *_args: Any, **_kwargs: Any) -> _CompatWebSocketServer:
			return _CompatWebSocketServer()

	websockets = _CompatWebSockets()
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .service import AICoreService
from .monitoring import ai_monitoring_system
from .ml_pipeline import ml_pipeline_framework
from .security import SecurityManager


class MessageType(str, Enum):
	"""Enumeration of WebSocket message types."""
	# Connection management
	CONNECT = "connect"
	DISCONNECT = "disconnect"
	HEARTBEAT = "heartbeat"

	# Authentication
	AUTHENTICATE = "authenticate"
	AUTH_SUCCESS = "auth_success"
	AUTH_FAILED = "auth_failed"

	# Subscriptions
	SUBSCRIBE = "subscribe"
	UNSUBSCRIBE = "unsubscribe"
	SUBSCRIPTION_CONFIRMED = "subscription_confirmed"

	# Monitoring events
	METRIC_UPDATE = "metric_update"
	ALERT_TRIGGERED = "alert_triggered"
	HEALTH_STATUS = "health_status"
	SYSTEM_EVENT = "system_event"

	# Pipeline events
	PIPELINE_STARTED = "pipeline_started"
	PIPELINE_STAGE_UPDATE = "pipeline_stage_update"
	PIPELINE_COMPLETED = "pipeline_completed"
	PIPELINE_FAILED = "pipeline_failed"

	# Inference events
	INFERENCE_STARTED = "inference_started"
	INFERENCE_PROGRESS = "inference_progress"
	INFERENCE_COMPLETED = "inference_completed"

	# Model events
	MODEL_DEPLOYED = "model_deployed"
	MODEL_UNDEPLOYED = "model_undeployed"
	MODEL_UPDATED = "model_updated"

	# Administrative events
	USER_JOINED = "user_joined"
	USER_LEFT = "user_left"
	ADMIN_MESSAGE = "admin_message"

	# Error events
	ERROR = "error"
	WARNING = "warning"


class SubscriptionType(str, Enum):
	"""Enumeration of WebSocket subscription types."""
	ALL_EVENTS = "all_events"
	SYSTEM_MONITORING = "system_monitoring"
	PIPELINE_EVENTS = "pipeline_events"
	INFERENCE_EVENTS = "inference_events"
	MODEL_EVENTS = "model_events"
	USER_EVENTS = "user_events"
	ADMIN_EVENTS = "admin_events"
	SPECIFIC_PIPELINE = "specific_pipeline"
	SPECIFIC_MODEL = "specific_model"


class WebSocketMessage(BaseModel):
	"""WebSocket message structure."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	message_id: str = Field(default_factory=uuid7str)
	message_type: MessageType = Field(..., description="Type of message")
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	sender_id: Optional[str] = Field(None, description="Sender identifier")
	target_id: Optional[str] = Field(None, description="Target identifier")
	subscription_type: Optional[SubscriptionType] = Field(None, description="Subscription type")
	data: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConnectionInfo(BaseModel):
	"""WebSocket connection information."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	connection_id: str = Field(default_factory=uuid7str)
	user_id: Optional[str] = Field(None, description="Authenticated user ID")
	user_name: Optional[str] = Field(None, description="User display name")
	user_roles: List[str] = Field(default_factory=list, description="User roles")
	connected_at: datetime = Field(default_factory=datetime.utcnow)
	last_activity: datetime = Field(default_factory=datetime.utcnow)
	subscriptions: Set[str] = Field(default_factory=set, description="Active subscriptions")
	client_info: Dict[str, Any] = Field(default_factory=dict, description="Client information")
	is_authenticated: bool = Field(default=False, description="Authentication status")


class WebSocketManager:
	"""Advanced WebSocket connection and message management system."""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		"""Initialize the WebSocket manager.

		Args:
			config: Optional configuration dictionary
		"""
		self.manager_id = uuid7str()
		self.config = config or {}
		self.connections: Dict[str, WebSocketServerProtocol] = {}
		self.connection_info: Dict[str, ConnectionInfo] = {}
		self.subscriptions: Dict[str, Set[str]] = {}  # subscription_type -> connection_ids
		self.message_handlers: Dict[MessageType, Callable] = {}
		self.security_manager = SecurityManager()
		self.ai_service = AICoreService()
		self.logger = logging.getLogger(__name__)
		self._cleanup_task: Optional[asyncio.Task] = None
		self._heartbeat_task: Optional[asyncio.Task] = None
		self._initialized = False

	async def initialize(self) -> None:
		"""Initialize the WebSocket manager."""
		try:
			await self.security_manager.initialize()
			await self.ai_service.initialize()

			# Register message handlers
			self._register_message_handlers()

			# Start background tasks
			self._cleanup_task = asyncio.create_task(self._cleanup_connections())
			self._heartbeat_task = asyncio.create_task(self._send_heartbeats())

			# Subscribe to monitoring events
			await self._subscribe_to_monitoring_events()

			self._initialized = True
			self._log_manager_event("WebSocket manager initialized successfully")

		except Exception as e:
			self._log_error(f"Failed to initialize WebSocket manager: {e}")
			raise

	async def handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
		"""Handle new WebSocket connection.

		Args:
			websocket: WebSocket connection
			path: Connection path
		"""
		connection_id = uuid7str()

		try:
			# Register connection
			self.connections[connection_id] = websocket
			self.connection_info[connection_id] = ConnectionInfo(
				connection_id=connection_id,
				client_info={
					"path": path,
					"remote_address": websocket.remote_address,
					"user_agent": websocket.request_headers.get("User-Agent", "")
				}
			)

			self._log_manager_event(
				f"New WebSocket connection: {connection_id}",
				{"remote_address": websocket.remote_address, "path": path}
			)

			# Send welcome message
			welcome_message = WebSocketMessage(
				message_type=MessageType.CONNECT,
				data={
					"connection_id": connection_id,
					"server_time": datetime.utcnow().isoformat(),
					"authentication_required": True,
					"available_subscriptions": [sub.value for sub in SubscriptionType]
				}
			)
			await self._send_message(connection_id, welcome_message)

			# Handle messages
			async for raw_message in websocket:
				await self._handle_message(connection_id, raw_message)

		except websockets.exceptions.ConnectionClosed:
			self._log_manager_event(f"WebSocket connection closed: {connection_id}")
		except Exception as e:
			self._log_error(f"Error handling WebSocket connection {connection_id}: {e}")
		finally:
			await self._cleanup_connection(connection_id)

	async def _handle_message(self, connection_id: str, raw_message: str) -> None:
		"""Handle incoming WebSocket message.

		Args:
			connection_id: Connection identifier
			raw_message: Raw message string
		"""
		try:
			# Parse message
			message_data = json.loads(raw_message)
			message = WebSocketMessage(**message_data)

			# Update last activity
			if connection_id in self.connection_info:
				self.connection_info[connection_id].last_activity = datetime.utcnow()

			# Route message to appropriate handler
			if message.message_type in self.message_handlers:
				await self.message_handlers[message.message_type](connection_id, message)
			else:
				self._log_warning(f"No handler for message type: {message.message_type}")

				error_message = WebSocketMessage(
					message_type=MessageType.ERROR,
					data={"error": f"Unknown message type: {message.message_type}"}
				)
				await self._send_message(connection_id, error_message)

		except json.JSONDecodeError:
			self._log_warning(f"Invalid JSON message from {connection_id}")

			error_message = WebSocketMessage(
				message_type=MessageType.ERROR,
				data={"error": "Invalid JSON format"}
			)
			await self._send_message(connection_id, error_message)

		except Exception as e:
			self._log_error(f"Error handling message from {connection_id}: {e}")

			error_message = WebSocketMessage(
				message_type=MessageType.ERROR,
				data={"error": f"Message processing failed: {str(e)}"}
			)
			await self._send_message(connection_id, error_message)

	async def _handle_authenticate(self, connection_id: str, message: WebSocketMessage) -> None:
		"""Handle authentication message."""
		try:
			token = message.data.get("token")
			if not token:
				auth_failed = WebSocketMessage(
					message_type=MessageType.AUTH_FAILED,
					data={"error": "Token required for authentication"}
				)
				await self._send_message(connection_id, auth_failed)
				return

			# Validate token
			user_info = await self.security_manager.validate_jwt_token(token)

			if user_info:
				# Update connection info
				conn_info = self.connection_info[connection_id]
				conn_info.user_id = user_info.get("user_id")
				conn_info.user_name = user_info.get("username")
				conn_info.user_roles = user_info.get("roles", [])
				conn_info.is_authenticated = True

				auth_success = WebSocketMessage(
					message_type=MessageType.AUTH_SUCCESS,
					data={
						"user_id": conn_info.user_id,
						"user_name": conn_info.user_name,
						"roles": conn_info.user_roles,
						"authenticated_at": datetime.utcnow().isoformat()
					}
				)
				await self._send_message(connection_id, auth_success)

				self._log_manager_event(
					f"User authenticated: {conn_info.user_name}",
					{"connection_id": connection_id, "user_id": conn_info.user_id}
				)
			else:
				auth_failed = WebSocketMessage(
					message_type=MessageType.AUTH_FAILED,
					data={"error": "Invalid token"}
				)
				await self._send_message(connection_id, auth_failed)

		except Exception as e:
			self._log_error(f"Authentication error for {connection_id}: {e}")

			auth_failed = WebSocketMessage(
				message_type=MessageType.AUTH_FAILED,
				data={"error": f"Authentication failed: {str(e)}"}
			)
			await self._send_message(connection_id, auth_failed)

	async def _handle_subscribe(self, connection_id: str, message: WebSocketMessage) -> None:
		"""Handle subscription message."""
		try:
			# Check authentication
			conn_info = self.connection_info.get(connection_id)
			if not conn_info or not conn_info.is_authenticated:
				error_message = WebSocketMessage(
					message_type=MessageType.ERROR,
					data={"error": "Authentication required for subscriptions"}
				)
				await self._send_message(connection_id, error_message)
				return

			subscription_type = message.data.get("subscription_type")
			if not subscription_type:
				error_message = WebSocketMessage(
					message_type=MessageType.ERROR,
					data={"error": "subscription_type required"}
				)
				await self._send_message(connection_id, error_message)
				return

			# Validate subscription type
			try:
				sub_type = SubscriptionType(subscription_type)
			except ValueError:
				error_message = WebSocketMessage(
					message_type=MessageType.ERROR,
					data={"error": f"Invalid subscription type: {subscription_type}"}
				)
				await self._send_message(connection_id, error_message)
				return

			# Check permissions
			if not await self._check_subscription_permission(conn_info, sub_type):
				error_message = WebSocketMessage(
					message_type=MessageType.ERROR,
					data={"error": f"Permission denied for subscription: {subscription_type}"}
				)
				await self._send_message(connection_id, error_message)
				return

			# Add subscription
			if subscription_type not in self.subscriptions:
				self.subscriptions[subscription_type] = set()

			self.subscriptions[subscription_type].add(connection_id)
			conn_info.subscriptions.add(subscription_type)

			# Send confirmation
			confirmation = WebSocketMessage(
				message_type=MessageType.SUBSCRIPTION_CONFIRMED,
				data={
					"subscription_type": subscription_type,
					"subscribed_at": datetime.utcnow().isoformat()
				}
			)
			await self._send_message(connection_id, confirmation)

			self._log_manager_event(
				f"Subscription added: {subscription_type}",
				{"connection_id": connection_id, "user_id": conn_info.user_id}
			)

		except Exception as e:
			self._log_error(f"Subscription error for {connection_id}: {e}")

			error_message = WebSocketMessage(
				message_type=MessageType.ERROR,
				data={"error": f"Subscription failed: {str(e)}"}
			)
			await self._send_message(connection_id, error_message)

	async def _handle_unsubscribe(self, connection_id: str, message: WebSocketMessage) -> None:
		"""Handle unsubscription message."""
		try:
			subscription_type = message.data.get("subscription_type")
			if not subscription_type:
				return

			# Remove subscription
			if subscription_type in self.subscriptions:
				self.subscriptions[subscription_type].discard(connection_id)

			conn_info = self.connection_info.get(connection_id)
			if conn_info:
				conn_info.subscriptions.discard(subscription_type)

			self._log_manager_event(
				f"Subscription removed: {subscription_type}",
				{"connection_id": connection_id}
			)

		except Exception as e:
			self._log_error(f"Unsubscription error for {connection_id}: {e}")

	async def _handle_heartbeat(self, connection_id: str, message: WebSocketMessage) -> None:
		"""Handle heartbeat message."""
		try:
			# Update last activity
			if connection_id in self.connection_info:
				self.connection_info[connection_id].last_activity = datetime.utcnow()

			# Send heartbeat response
			heartbeat_response = WebSocketMessage(
				message_type=MessageType.HEARTBEAT,
				data={
					"server_time": datetime.utcnow().isoformat(),
					"connection_active": True
				}
			)
			await self._send_message(connection_id, heartbeat_response)

		except Exception as e:
			self._log_error(f"Heartbeat error for {connection_id}: {e}")

	async def broadcast_event(
		self,
		event_type: MessageType,
		data: Dict[str, Any],
		subscription_filter: Optional[SubscriptionType] = None,
		user_filter: Optional[Callable[[ConnectionInfo], bool]] = None
	) -> None:
		"""Broadcast event to subscribers.

		Args:
			event_type: Type of event to broadcast
			data: Event data
			subscription_filter: Optional subscription type filter
			user_filter: Optional user filter function
		"""
		try:
			message = WebSocketMessage(
				message_type=event_type,
				data=data,
				timestamp=datetime.utcnow()
			)

			# Determine target connections
			target_connections = set()

			if subscription_filter:
				# Send to specific subscription type
				target_connections.update(
					self.subscriptions.get(subscription_filter.value, set())
				)
			else:
				# Send to all connections
				target_connections.update(self.connections.keys())

			# Apply user filter if provided
			if user_filter:
				filtered_connections = set()
				for conn_id in target_connections:
					conn_info = self.connection_info.get(conn_id)
					if conn_info and user_filter(conn_info):
						filtered_connections.add(conn_id)
				target_connections = filtered_connections

			# Send to target connections
			send_tasks = []
			for connection_id in target_connections:
				if connection_id in self.connections:
					task = self._send_message(connection_id, message)
					send_tasks.append(task)

			if send_tasks:
				await asyncio.gather(*send_tasks, return_exceptions=True)

			self._log_manager_event(
				f"Event broadcasted: {event_type.value}",
				{"target_connections": len(target_connections)}
			)

		except Exception as e:
			self._log_error(f"Error broadcasting event {event_type}: {e}")

	async def _send_message(self, connection_id: str, message: WebSocketMessage) -> None:
		"""Send message to specific connection.

		Args:
			connection_id: Target connection ID
			message: Message to send
		"""
		try:
			websocket = self.connections.get(connection_id)
			if websocket and not websocket.closed:
				message_json = json.dumps(message.model_dump(), default=str)
				await websocket.send(message_json)

		except websockets.exceptions.ConnectionClosed:
			await self._cleanup_connection(connection_id)
		except Exception as e:
			self._log_error(f"Error sending message to {connection_id}: {e}")
			await self._cleanup_connection(connection_id)

	async def _cleanup_connection(self, connection_id: str) -> None:
		"""Clean up connection resources.

		Args:
			connection_id: Connection to clean up
		"""
		try:
			# Remove from connections
			if connection_id in self.connections:
				del self.connections[connection_id]

			# Remove from subscriptions
			conn_info = self.connection_info.get(connection_id)
			if conn_info:
				for subscription_type in conn_info.subscriptions:
					if subscription_type in self.subscriptions:
						self.subscriptions[subscription_type].discard(connection_id)

			# Remove connection info
			if connection_id in self.connection_info:
				del self.connection_info[connection_id]

			self._log_manager_event(f"Connection cleaned up: {connection_id}")

		except Exception as e:
			self._log_error(f"Error cleaning up connection {connection_id}: {e}")

	async def _cleanup_connections(self) -> None:
		"""Periodic cleanup of stale connections."""
		while True:
			try:
				current_time = datetime.utcnow()
				timeout_threshold = current_time - timedelta(minutes=30)  # 30 minute timeout

				stale_connections = []
				for connection_id, conn_info in self.connection_info.items():
					if conn_info.last_activity < timeout_threshold:
						stale_connections.append(connection_id)

				# Clean up stale connections
				for connection_id in stale_connections:
					await self._cleanup_connection(connection_id)

				if stale_connections:
					self._log_manager_event(
						f"Cleaned up {len(stale_connections)} stale connections"
					)

				await asyncio.sleep(300)  # Run every 5 minutes

			except asyncio.CancelledError:
				break
			except Exception as e:
				self._log_error(f"Error in connection cleanup: {e}")
				await asyncio.sleep(300)

	async def _send_heartbeats(self) -> None:
		"""Send periodic heartbeats to all connections."""
		while True:
			try:
				if self.connections:
					heartbeat_message = WebSocketMessage(
						message_type=MessageType.HEARTBEAT,
						data={
							"server_time": datetime.utcnow().isoformat(),
							"active_connections": len(self.connections)
						}
					)

					# Send heartbeat to all connections
					send_tasks = []
					for connection_id in list(self.connections.keys()):
						task = self._send_message(connection_id, heartbeat_message)
						send_tasks.append(task)

					if send_tasks:
						await asyncio.gather(*send_tasks, return_exceptions=True)

				await asyncio.sleep(30)  # Send heartbeat every 30 seconds

			except asyncio.CancelledError:
				break
			except Exception as e:
				self._log_error(f"Error sending heartbeats: {e}")
				await asyncio.sleep(30)

	async def _subscribe_to_monitoring_events(self) -> None:
		"""Subscribe to monitoring system events."""
		try:
			# This would integrate with the monitoring system to receive events
			# For now, we'll set up a periodic task to check for events
			asyncio.create_task(self._monitor_system_events())

		except Exception as e:
			self._log_error(f"Error subscribing to monitoring events: {e}")

	async def _monitor_system_events(self) -> None:
		"""Monitor system events and broadcast to subscribers."""
		while True:
			try:
				# Get system health
				if ai_monitoring_system._initialized:
					health_data = await ai_monitoring_system.get_system_health()

					await self.broadcast_event(
						MessageType.HEALTH_STATUS,
						health_data,
						subscription_filter=SubscriptionType.SYSTEM_MONITORING
					)

				# Check for pipeline events
				if ml_pipeline_framework._initialized:
					# Get recent executions and broadcast updates
					for execution in ml_pipeline_framework.orchestrator.executions.values():
						if execution.status == "running":
							await self.broadcast_event(
								MessageType.PIPELINE_STAGE_UPDATE,
								{
									"execution_id": execution.execution_id,
									"pipeline_id": execution.pipeline_id,
									"current_stage": execution.current_stage,
									"status": execution.status
								},
								subscription_filter=SubscriptionType.PIPELINE_EVENTS
							)

				await asyncio.sleep(10)  # Check every 10 seconds

			except asyncio.CancelledError:
				break
			except Exception as e:
				self._log_error(f"Error monitoring system events: {e}")
				await asyncio.sleep(10)

	async def _check_subscription_permission(
		self,
		conn_info: ConnectionInfo,
		subscription_type: SubscriptionType
	) -> bool:
		"""Check if user has permission for subscription type.

		Args:
			conn_info: Connection information
			subscription_type: Requested subscription type

		Returns:
			bool: True if permission granted
		"""
		# Basic permission checks
		if subscription_type == SubscriptionType.ADMIN_EVENTS:
			return "admin" in conn_info.user_roles

		if subscription_type == SubscriptionType.ALL_EVENTS:
			return "admin" in conn_info.user_roles or "operator" in conn_info.user_roles

		# Default: allow for authenticated users
		return conn_info.is_authenticated

	def _register_message_handlers(self) -> None:
		"""Register WebSocket message handlers."""
		self.message_handlers = {
			MessageType.AUTHENTICATE: self._handle_authenticate,
			MessageType.SUBSCRIBE: self._handle_subscribe,
			MessageType.UNSUBSCRIBE: self._handle_unsubscribe,
			MessageType.HEARTBEAT: self._handle_heartbeat
		}

	def _log_manager_event(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log manager events with structured context."""
		self.logger.info(f"[WebSocketManager] {message}", extra=context or {})

	def _log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log warning messages with structured context."""
		self.logger.warning(f"[WebSocketManager] {message}", extra=context or {})

	def _log_error(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log error messages with structured context."""
		self.logger.error(f"[WebSocketManager] {message}", extra=context or {})


class WebSocketServer:
	"""WebSocket server for real-time AICR communications."""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		"""Initialize the WebSocket server.

		Args:
			config: Optional configuration dictionary
		"""
		self.server_id = uuid7str()
		self.config = config or {}
		self.manager = WebSocketManager(config)
		self.server: Optional[websockets.WebSocketServer] = None
		self.logger = logging.getLogger(__name__)
		self._initialized = False

	async def initialize(self) -> None:
		"""Initialize the WebSocket server."""
		try:
			await self.manager.initialize()
			self._initialized = True
			self._log_server_event("WebSocket server initialized successfully")

		except Exception as e:
			self._log_error(f"Failed to initialize WebSocket server: {e}")
			raise

	async def start_server(
		self,
		host: str = "localhost",
		port: int = 8765
	) -> None:
		"""Start the WebSocket server.

		Args:
			host: Server host
			port: Server port
		"""
		if not self._initialized:
			raise RuntimeError("Server not initialized")

		try:
			self.server = await websockets.serve(
				self.manager.handle_connection,
				host,
				port,
				ping_interval=30,
				ping_timeout=10,
				max_size=1024*1024,  # 1MB max message size
				max_queue=100
			)

			self._log_server_event(
				f"WebSocket server started on {host}:{port}",
				{"host": host, "port": port}
			)

		except Exception as e:
			self._log_error(f"Failed to start WebSocket server: {e}")
			raise

	async def stop_server(self) -> None:
		"""Stop the WebSocket server."""
		try:
			if self.server:
				self.server.close()
				await self.server.wait_closed()
				self.server = None

			self._log_server_event("WebSocket server stopped")

		except Exception as e:
			self._log_error(f"Error stopping WebSocket server: {e}")

	async def broadcast_event(
		self,
		event_type: MessageType,
		data: Dict[str, Any],
		subscription_filter: Optional[SubscriptionType] = None
	) -> None:
		"""Broadcast event to connected clients.

		Args:
			event_type: Type of event
			data: Event data
			subscription_filter: Optional subscription filter
		"""
		if self._initialized:
			await self.manager.broadcast_event(event_type, data, subscription_filter)

	def get_connection_stats(self) -> Dict[str, Any]:
		"""Get connection statistics.

		Returns:
			Dict[str, Any]: Connection statistics
		"""
		authenticated_connections = sum(
			1 for conn_info in self.manager.connection_info.values()
			if conn_info.is_authenticated
		)

		subscription_stats = {
			sub_type: len(connections)
			for sub_type, connections in self.manager.subscriptions.items()
		}

		return {
			"total_connections": len(self.manager.connections),
			"authenticated_connections": authenticated_connections,
			"subscription_stats": subscription_stats,
			"server_uptime": datetime.utcnow().isoformat(),
			"server_id": self.server_id
		}

	def _log_server_event(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log server events with structured context."""
		self.logger.info(f"[WebSocketServer] {message}", extra=context or {})

	def _log_error(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log error messages with structured context."""
		self.logger.error(f"[WebSocketServer] {message}", extra=context or {})


# Global WebSocket server instance
websocket_server = WebSocketServer()

# Export key classes and functions
__all__ = [
	"WebSocketServer",
	"WebSocketManager",
	"WebSocketMessage",
	"ConnectionInfo",
	"MessageType",
	"SubscriptionType",
	"websocket_server"
]

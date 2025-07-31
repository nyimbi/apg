"""
Real-Time Collaboration WebSocket Manager

Handles WebSocket connections, presence tracking, and real-time message routing
for Flask-AppBuilder page-level collaboration with APG integration.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from uuid_extensions import uuid7str
import logging
from dataclasses import dataclass
from enum import Enum

# Import WebRTC signaling for integration
try:
	from .webrtc_signaling import webrtc_signaling, handle_webrtc_message
except ImportError:
	webrtc_signaling = None
	handle_webrtc_message = None
	logging.warning("WebRTC signaling not available, WebRTC features disabled")

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedError


class ConnectionStatus(Enum):
	"""WebSocket connection status"""
	CONNECTING = "connecting"
	CONNECTED = "connected"
	DISCONNECTED = "disconnected"
	ERROR = "error"


class MessageType(Enum):
	"""Real-time message types"""
	# Connection management
	CONNECT = "connect"
	DISCONNECT = "disconnect"
	HEARTBEAT = "heartbeat"
	
	# Presence tracking
	PRESENCE_UPDATE = "presence_update"
	USER_JOIN = "user_join"
	USER_LEAVE = "user_leave"
	CURSOR_POSITION = "cursor_position"
	
	# Flask-AppBuilder page collaboration
	PAGE_JOIN = "page_join"
	PAGE_LEAVE = "page_leave"
	FORM_FIELD_FOCUS = "form_field_focus"
	FORM_FIELD_BLUR = "form_field_blur"
	FORM_FIELD_CHANGE = "form_field_change"
	FORM_DELEGATION = "form_delegation"
	ASSISTANCE_REQUEST = "assistance_request"
	
	# Chat and messaging
	CHAT_MESSAGE = "chat_message"
	TYPING_START = "typing_start"
	TYPING_STOP = "typing_stop"
	
	# Video/Audio collaboration
	VIDEO_CALL_START = "video_call_start"
	VIDEO_CALL_END = "video_call_end"
	SCREEN_SHARE_START = "screen_share_start"
	SCREEN_SHARE_END = "screen_share_end"
	AUDIO_TOGGLE = "audio_toggle"
	VIDEO_TOGGLE = "video_toggle"
	HAND_RAISE = "hand_raise"
	REACTION = "reaction"
	
	# Annotations and markup
	ANNOTATION_ADD = "annotation_add"
	ANNOTATION_REMOVE = "annotation_remove"
	ANNOTATION_UPDATE = "annotation_update"
	
	# Error handling
	ERROR = "error"
	VALIDATION_ERROR = "validation_error"


@dataclass
class WebSocketConnection:
	"""WebSocket connection metadata"""
	connection_id: str
	websocket: WebSocketServerProtocol
	user_id: str
	tenant_id: str
	page_url: str
	session_id: str | None
	connected_at: datetime
	last_heartbeat: datetime
	status: ConnectionStatus
	user_agent: str | None = None
	ip_address: str | None = None
	presence_data: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.presence_data is None:
			self.presence_data = {}


@dataclass
class PresenceInfo:
	"""User presence information"""
	user_id: str
	display_name: str
	page_url: str
	cursor_position: Dict[str, Any] | None
	active_field: str | None
	status: str  # active, away, busy
	last_activity: datetime
	is_typing: bool = False
	video_enabled: bool = False
	audio_enabled: bool = False


class WebSocketManager:
	"""
	Manages WebSocket connections for real-time collaboration.
	
	Handles connection lifecycle, presence tracking, message routing,
	and Flask-AppBuilder page-level collaboration features.
	"""
	
	def __init__(self):
		# Connection management
		self._connections: Dict[str, WebSocketConnection] = {}
		self._user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
		self._page_connections: Dict[str, Set[str]] = {}  # page_url -> connection_ids
		self._session_connections: Dict[str, Set[str]] = {}  # session_id -> connection_ids
		
		# Presence tracking
		self._presence_info: Dict[str, PresenceInfo] = {}  # user_id -> presence
		self._page_presence: Dict[str, Set[str]] = {}  # page_url -> user_ids
		
		# Message routing and handlers
		self._message_handlers: Dict[MessageType, callable] = {}
		self._setup_message_handlers()
		
		# Background tasks
		self._heartbeat_task: asyncio.Task | None = None
		self._cleanup_task: asyncio.Task | None = None
		
		# Configuration
		self.heartbeat_interval = 30  # seconds
		self.connection_timeout = 300  # seconds
		self.max_connections_per_user = 10
		
		# Logging
		self._logger = logging.getLogger(__name__)
	
	def _setup_message_handlers(self):
		"""Setup message type handlers"""
		self._message_handlers = {
			MessageType.CONNECT: self._handle_connect,
			MessageType.DISCONNECT: self._handle_disconnect,
			MessageType.HEARTBEAT: self._handle_heartbeat,
			MessageType.PRESENCE_UPDATE: self._handle_presence_update,
			MessageType.PAGE_JOIN: self._handle_page_join,
			MessageType.PAGE_LEAVE: self._handle_page_leave,
			MessageType.FORM_FIELD_FOCUS: self._handle_form_field_focus,
			MessageType.FORM_FIELD_BLUR: self._handle_form_field_blur,
			MessageType.FORM_FIELD_CHANGE: self._handle_form_field_change,
			MessageType.FORM_DELEGATION: self._handle_form_delegation,
			MessageType.ASSISTANCE_REQUEST: self._handle_assistance_request,
			MessageType.CHAT_MESSAGE: self._handle_chat_message,
			MessageType.TYPING_START: self._handle_typing_start,
			MessageType.TYPING_STOP: self._handle_typing_stop,
			MessageType.CURSOR_POSITION: self._handle_cursor_position,
			MessageType.VIDEO_CALL_START: self._handle_video_call_start,
			MessageType.SCREEN_SHARE_START: self._handle_screen_share_start,
			MessageType.ANNOTATION_ADD: self._handle_annotation_add,
		}
	
	async def start(self) -> None:
		"""Start the WebSocket manager and background tasks"""
		self._logger.info("Starting WebSocket manager")
		
		# Start background tasks
		self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
		self._cleanup_task = asyncio.create_task(self._cleanup_loop())
		
		self._logger.info("WebSocket manager started successfully")
	
	async def stop(self) -> None:
		"""Stop the WebSocket manager and cleanup"""
		self._logger.info("Stopping WebSocket manager")
		
		# Cancel background tasks
		if self._heartbeat_task:
			self._heartbeat_task.cancel()
		if self._cleanup_task:
			self._cleanup_task.cancel()
		
		# Close all connections
		for connection in list(self._connections.values()):
			await self._close_connection(connection.connection_id, "Server shutdown")
		
		self._logger.info("WebSocket manager stopped")
	
	async def handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
		"""Handle new WebSocket connection"""
		connection_id = uuid7str()
		
		try:
			# Extract connection info from path/headers
			user_id, tenant_id, page_url = await self._extract_connection_info(websocket, path)
			
			# Validate connection
			if not await self._validate_connection(user_id, tenant_id):
				await websocket.close(code=4001, reason="Unauthorized")
				return
			
			# Create connection
			connection = WebSocketConnection(
				connection_id=connection_id,
				websocket=websocket,
				user_id=user_id,
				tenant_id=tenant_id,
				page_url=page_url,
				session_id=None,
				connected_at=datetime.utcnow(),
				last_heartbeat=datetime.utcnow(),
				status=ConnectionStatus.CONNECTED,
				user_agent=websocket.request_headers.get('user-agent'),
				ip_address=websocket.remote_address[0] if websocket.remote_address else None
			)
			
			# Register connection
			await self._register_connection(connection)
			
			# Handle messages
			await self._handle_connection_messages(connection)
			
		except Exception as e:
			self._logger.error(f"Error handling connection {connection_id}: {e}")
		finally:
			if connection_id in self._connections:
				await self._close_connection(connection_id, "Connection ended")
	
	async def _extract_connection_info(self, websocket: WebSocketServerProtocol, path: str) -> tuple[str, str, str]:
		"""Extract user_id, tenant_id, and page_url from connection"""
		# Parse path: /ws/rtc/{tenant_id}/{user_id}?page_url={encoded_url}
		# This would integrate with APG auth_rbac for authentication
		
		# For now, return mock data - would be replaced with real auth
		return "user123", "tenant123", "http://localhost:5000/some/page"
	
	async def _validate_connection(self, user_id: str, tenant_id: str) -> bool:
		"""Validate connection with APG auth_rbac"""
		# Integration point with APG auth_rbac capability
		# Would validate JWT tokens, permissions, etc.
		return True
	
	async def _register_connection(self, connection: WebSocketConnection) -> None:
		"""Register new connection"""
		conn_id = connection.connection_id
		user_id = connection.user_id
		page_url = connection.page_url
		
		# Store connection
		self._connections[conn_id] = connection
		
		# Update user connections
		if user_id not in self._user_connections:
			self._user_connections[user_id] = set()
		self._user_connections[user_id].add(conn_id)
		
		# Update page connections
		if page_url not in self._page_connections:
			self._page_connections[page_url] = set()
		self._page_connections[page_url].add(conn_id)
		
		# Create/update presence
		await self._update_user_presence(connection)
		
		# Notify other users on the same page
		await self._broadcast_to_page(page_url, {
			'type': MessageType.USER_JOIN.value,
			'user_id': user_id,
			'page_url': page_url,
			'timestamp': datetime.utcnow().isoformat()
		}, exclude_connections={conn_id})
		
		self._logger.info(f"Registered connection {conn_id} for user {user_id} on page {page_url}")
	
	async def _close_connection(self, connection_id: str, reason: str = None) -> None:
		"""Close and cleanup connection"""
		if connection_id not in self._connections:
			return
		
		connection = self._connections[connection_id]
		user_id = connection.user_id
		page_url = connection.page_url
		
		# Remove from tracking
		del self._connections[connection_id]
		
		if user_id in self._user_connections:
			self._user_connections[user_id].discard(connection_id)
			if not self._user_connections[user_id]:
				del self._user_connections[user_id]
		
		if page_url in self._page_connections:
			self._page_connections[page_url].discard(connection_id)
			if not self._page_connections[page_url]:
				del self._page_connections[page_url]
		
		# Update presence
		await self._remove_user_presence(user_id, page_url)
		
		# Close WebSocket
		try:
			if not connection.websocket.closed:
				await connection.websocket.close(reason=reason)
		except Exception as e:
			self._logger.warning(f"Error closing WebSocket: {e}")
		
		# Notify other users
		await self._broadcast_to_page(page_url, {
			'type': MessageType.USER_LEAVE.value,
			'user_id': user_id,
			'page_url': page_url,
			'timestamp': datetime.utcnow().isoformat()
		})
		
		self._logger.info(f"Closed connection {connection_id} for user {user_id}: {reason}")
	
	async def _handle_connection_messages(self, connection: WebSocketConnection) -> None:
		"""Handle messages for a connection"""
		try:
			async for message in connection.websocket:
				try:
					# Parse message
					data = json.loads(message)
					message_type_str = data.get('type', '')
					
					# Update heartbeat
					connection.last_heartbeat = datetime.utcnow()
					
					# Handle WebRTC messages separately
					if message_type_str.startswith('webrtc_') and handle_webrtc_message:
						try:
							response = await handle_webrtc_message(connection.user_id, data)
							if response and 'error' not in response:
								await self._send_message(connection, {
									'type': 'webrtc_response',
									'original_type': message_type_str,
									'response': response
								})
							elif response and 'error' in response:
								await self._send_error(connection, f"WebRTC error: {response['error']}")
						except Exception as e:
							self._logger.error(f"WebRTC message handling error: {e}")
							await self._send_error(connection, "WebRTC processing failed")
						continue
					
					# Handle regular messages
					try:
						message_type = MessageType(message_type_str)
						# Route message to handler
						if message_type in self._message_handlers:
							await self._message_handlers[message_type](connection, data)
						else:
							self._logger.warning(f"Unknown message type: {message_type}")
					except ValueError:
						self._logger.warning(f"Unknown message type: {message_type_str}")
				
				except json.JSONDecodeError:
					await self._send_error(connection, "Invalid JSON message")
					await self._send_error(connection, "Unknown message type")
				except Exception as e:
					self._logger.error(f"Error handling message: {e}")
					await self._send_error(connection, "Internal server error")
		
		except ConnectionClosed:
			self._logger.info(f"Connection {connection.connection_id} closed normally")
		except ConnectionClosedError as e:
			self._logger.warning(f"Connection {connection.connection_id} closed with error: {e}")
		except Exception as e:
			self._logger.error(f"Unexpected error in message handler: {e}")
	
	# Message handlers
	async def _handle_connect(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle connection message"""
		await self._send_message(connection, {
			'type': MessageType.CONNECT.value,
			'connection_id': connection.connection_id,
			'status': 'connected',
			'timestamp': datetime.utcnow().isoformat()
		})
	
	async def _handle_disconnect(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle disconnect message"""
		await self._close_connection(connection.connection_id, "Client disconnect")
	
	async def _handle_heartbeat(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle heartbeat message"""
		connection.last_heartbeat = datetime.utcnow()
		await self._send_message(connection, {
			'type': MessageType.HEARTBEAT.value,
			'timestamp': datetime.utcnow().isoformat()
		})
	
	async def _handle_presence_update(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle presence update"""
		user_id = connection.user_id
		
		if user_id in self._presence_info:
			presence = self._presence_info[user_id]
			presence.status = data.get('status', presence.status)
			presence.last_activity = datetime.utcnow()
			
			# Broadcast presence update
			await self._broadcast_to_page(connection.page_url, {
				'type': MessageType.PRESENCE_UPDATE.value,
				'user_id': user_id,
				'status': presence.status,
				'timestamp': datetime.utcnow().isoformat()
			}, exclude_connections={connection.connection_id})
	
	async def _handle_page_join(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle Flask-AppBuilder page join"""
		page_url = data.get('page_url', connection.page_url)
		form_data = data.get('form_data', {})
		record_id = data.get('record_id')
		
		# Update connection page context
		connection.page_url = page_url
		
		# Store form context for collaboration
		connection.presence_data.update({
			'form_data': form_data,
			'record_id': record_id,
			'page_type': data.get('page_type', 'unknown')
		})
		
		# Notify APG page collaboration system
		# This would integrate with RTCPageCollaboration model
		
		await self._send_message(connection, {
			'type': MessageType.PAGE_JOIN.value,
			'page_url': page_url,
			'current_users': list(self._page_presence.get(page_url, set())),
			'timestamp': datetime.utcnow().isoformat()
		})
	
	async def _handle_form_field_focus(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle form field focus for collaborative editing"""
		field_name = data.get('field_name')
		
		if field_name:
			# Update presence
			if connection.user_id in self._presence_info:
				self._presence_info[connection.user_id].active_field = field_name
			
			# Broadcast field lock to other users
			await self._broadcast_to_page(connection.page_url, {
				'type': MessageType.FORM_FIELD_FOCUS.value,
				'user_id': connection.user_id,
				'field_name': field_name,
				'timestamp': datetime.utcnow().isoformat()
			}, exclude_connections={connection.connection_id})
	
	async def _handle_form_field_blur(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle form field blur"""
		field_name = data.get('field_name')
		
		if field_name:
			# Update presence
			if connection.user_id in self._presence_info:
				self._presence_info[connection.user_id].active_field = None
			
			# Broadcast field unlock
			await self._broadcast_to_page(connection.page_url, {
				'type': MessageType.FORM_FIELD_BLUR.value,
				'user_id': connection.user_id,
				'field_name': field_name,
				'timestamp': datetime.utcnow().isoformat()
			}, exclude_connections={connection.connection_id})
	
	async def _handle_form_field_change(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle real-time form field changes"""
		field_name = data.get('field_name')
		field_value = data.get('field_value')
		
		# Broadcast change to other users for real-time collaboration
		await self._broadcast_to_page(connection.page_url, {
			'type': MessageType.FORM_FIELD_CHANGE.value,
			'user_id': connection.user_id,
			'field_name': field_name,
			'field_value': field_value,
			'timestamp': datetime.utcnow().isoformat()
		}, exclude_connections={connection.connection_id})
	
	async def _handle_form_delegation(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle form field delegation"""
		field_name = data.get('field_name')
		delegatee_id = data.get('delegatee_id')
		instructions = data.get('instructions', '')
		
		# Find delegatee connections
		delegatee_connections = self._user_connections.get(delegatee_id, set())
		
		for conn_id in delegatee_connections:
			if conn_id in self._connections:
				await self._send_message(self._connections[conn_id], {
					'type': MessageType.FORM_DELEGATION.value,
					'delegator_id': connection.user_id,
					'field_name': field_name,
					'instructions': instructions,
					'page_url': connection.page_url,
					'timestamp': datetime.utcnow().isoformat()
				})
	
	async def _handle_assistance_request(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle assistance request"""
		field_name = data.get('field_name')
		description = data.get('description', '')
		
		# Broadcast assistance request to all users on the page
		await self._broadcast_to_page(connection.page_url, {
			'type': MessageType.ASSISTANCE_REQUEST.value,
			'requester_id': connection.user_id,
			'field_name': field_name,
			'description': description,
			'page_url': connection.page_url,
			'timestamp': datetime.utcnow().isoformat()
		}, exclude_connections={connection.connection_id})
	
	async def _handle_chat_message(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle chat message"""
		message = data.get('message', '')
		message_type = data.get('message_type', 'text')
		
		# Broadcast chat message to page
		await self._broadcast_to_page(connection.page_url, {
			'type': MessageType.CHAT_MESSAGE.value,
			'user_id': connection.user_id,
			'message': message,
			'message_type': message_type,
			'page_url': connection.page_url,
			'timestamp': datetime.utcnow().isoformat()
		})
	
	async def _handle_typing_start(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle typing start indicator"""
		if connection.user_id in self._presence_info:
			self._presence_info[connection.user_id].is_typing = True
		
		await self._broadcast_to_page(connection.page_url, {
			'type': MessageType.TYPING_START.value,
			'user_id': connection.user_id,
			'timestamp': datetime.utcnow().isoformat()
		}, exclude_connections={connection.connection_id})
	
	async def _handle_typing_stop(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle typing stop indicator"""
		if connection.user_id in self._presence_info:
			self._presence_info[connection.user_id].is_typing = False
		
		await self._broadcast_to_page(connection.page_url, {
			'type': MessageType.TYPING_STOP.value,
			'user_id': connection.user_id,
			'timestamp': datetime.utcnow().isoformat()
		}, exclude_connections={connection.connection_id})
	
	async def _handle_cursor_position(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle cursor position updates"""
		cursor_data = data.get('cursor', {})
		
		if connection.user_id in self._presence_info:
			self._presence_info[connection.user_id].cursor_position = cursor_data
		
		# Broadcast cursor position (throttled for performance)
		await self._broadcast_to_page(connection.page_url, {
			'type': MessageType.CURSOR_POSITION.value,
			'user_id': connection.user_id,
			'cursor': cursor_data,
			'timestamp': datetime.utcnow().isoformat()
		}, exclude_connections={connection.connection_id})
	
	async def _handle_video_call_start(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle video call start"""
		call_id = data.get('call_id')
		
		# Update presence
		if connection.user_id in self._presence_info:
			self._presence_info[connection.user_id].video_enabled = True
		
		# Broadcast to page
		await self._broadcast_to_page(connection.page_url, {
			'type': MessageType.VIDEO_CALL_START.value,
			'user_id': connection.user_id,
			'call_id': call_id,
			'timestamp': datetime.utcnow().isoformat()
		})
	
	async def _handle_screen_share_start(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle screen share start"""
		share_id = data.get('share_id')
		share_type = data.get('share_type', 'desktop')
		
		# Broadcast to page
		await self._broadcast_to_page(connection.page_url, {
			'type': MessageType.SCREEN_SHARE_START.value,
			'user_id': connection.user_id,
			'share_id': share_id,
			'share_type': share_type,
			'timestamp': datetime.utcnow().isoformat()
		})
	
	async def _handle_annotation_add(self, connection: WebSocketConnection, data: Dict[str, Any]) -> None:
		"""Handle page annotation addition"""
		annotation = data.get('annotation', {})
		
		# Broadcast annotation to page
		await self._broadcast_to_page(connection.page_url, {
			'type': MessageType.ANNOTATION_ADD.value,
			'user_id': connection.user_id,
			'annotation': annotation,
			'timestamp': datetime.utcnow().isoformat()
		}, exclude_connections={connection.connection_id})
	
	# Utility methods
	async def _update_user_presence(self, connection: WebSocketConnection) -> None:
		"""Update user presence information"""
		user_id = connection.user_id
		page_url = connection.page_url
		
		# Create or update presence
		self._presence_info[user_id] = PresenceInfo(
			user_id=user_id,
			display_name=f"User {user_id}",  # Would get from APG user management
			page_url=page_url,
			cursor_position=None,
			active_field=None,
			status="active",
			last_activity=datetime.utcnow()
		)
		
		# Update page presence
		if page_url not in self._page_presence:
			self._page_presence[page_url] = set()
		self._page_presence[page_url].add(user_id)
	
	async def _remove_user_presence(self, user_id: str, page_url: str) -> None:
		"""Remove user from presence tracking"""
		# Remove from presence if no other connections for this user
		if user_id not in self._user_connections:
			self._presence_info.pop(user_id, None)
		
		# Remove from page presence
		if page_url in self._page_presence:
			self._page_presence[page_url].discard(user_id)
			if not self._page_presence[page_url]:
				del self._page_presence[page_url]
	
	async def _broadcast_to_page(self, page_url: str, message: Dict[str, Any], exclude_connections: Set[str] = None) -> None:
		"""Broadcast message to all connections on a page"""
		if exclude_connections is None:
			exclude_connections = set()
		
		if page_url not in self._page_connections:
			return
		
		message_json = json.dumps(message)
		
		for conn_id in self._page_connections[page_url]:
			if conn_id in exclude_connections or conn_id not in self._connections:
				continue
			
			try:
				connection = self._connections[conn_id]
				await connection.websocket.send(message_json)
			except Exception as e:
				self._logger.warning(f"Failed to send message to connection {conn_id}: {e}")
				# Connection probably dead, will be cleaned up in background
	
	async def _send_message(self, connection: WebSocketConnection, message: Dict[str, Any]) -> None:
		"""Send message to specific connection"""
		try:
			await connection.websocket.send(json.dumps(message))
		except Exception as e:
			self._logger.warning(f"Failed to send message to connection {connection.connection_id}: {e}")
	
	async def _send_error(self, connection: WebSocketConnection, error_message: str) -> None:
		"""Send error message to connection"""
		await self._send_message(connection, {
			'type': MessageType.ERROR.value,
			'error': error_message,
			'timestamp': datetime.utcnow().isoformat()
		})
	
	async def _heartbeat_loop(self) -> None:
		"""Background heartbeat monitoring"""
		while True:
			try:
				await asyncio.sleep(self.heartbeat_interval)
				
				current_time = datetime.utcnow()
				timeout_threshold = current_time - timedelta(seconds=self.connection_timeout)
				
				# Find stale connections
				stale_connections = []
				for conn_id, connection in self._connections.items():
					if connection.last_heartbeat < timeout_threshold:
						stale_connections.append(conn_id)
				
				# Close stale connections
				for conn_id in stale_connections:
					await self._close_connection(conn_id, "Connection timeout")
				
				if stale_connections:
					self._logger.info(f"Cleaned up {len(stale_connections)} stale connections")
			
			except asyncio.CancelledError:
				break
			except Exception as e:
				self._logger.error(f"Error in heartbeat loop: {e}")
	
	async def _cleanup_loop(self) -> None:
		"""Background cleanup of stale data"""
		while True:
			try:
				await asyncio.sleep(300)  # 5 minutes
				
				# Cleanup would include:
				# - Remove stale presence data
				# - Clean up empty page tracking
				# - Persist session data to database
				# - Update APG analytics
				
				self._logger.debug("Performed background cleanup")
			
			except asyncio.CancelledError:
				break
			except Exception as e:
				self._logger.error(f"Error in cleanup loop: {e}")
	
	async def _handle_page_leave(self, message: Dict[str, Any], connection: WebSocketConnection) -> None:
		"""Handle user leaving a page"""
		page_url = message.get('page_url')
		if not page_url:
			return
		
		# Remove connection from page tracking
		if page_url in self._page_connections:
			self._page_connections[page_url].discard(connection.connection_id)
			
			# Clean up empty page tracking
			if not self._page_connections[page_url]:
				del self._page_connections[page_url]
		
		# Update presence info
		if connection.user_id:
			user_pages = self._presence_info.get(connection.user_id, {}).get('pages', [])
			if page_url in user_pages:
				user_pages.remove(page_url)
		
		# Broadcast user left page
		await self._broadcast_to_page(page_url, {
			'type': MessageType.USER_LEAVE.value,
			'user_id': connection.user_id,
			'page_url': page_url,
			'timestamp': datetime.utcnow().isoformat()
		}, exclude_connection=connection.connection_id)
		
		self._logger.debug(f"User {connection.user_id} left page {page_url}")
	
	def get_connection_stats(self) -> Dict[str, Any]:
		"""Get current connection statistics"""
		return {
			'total_connections': len(self._connections),
			'total_users': len(self._user_connections),
			'total_pages': len(self._page_connections),
			'active_sessions': len(self._session_connections),
			'presence_users': len(self._presence_info)
		}


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
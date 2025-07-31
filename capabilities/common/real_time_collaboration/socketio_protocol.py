"""
Socket.IO Protocol Implementation for APG Real-Time Collaboration

Provides Socket.IO communication with enhanced WebSocket fallbacks,
cross-browser compatibility, and real-time bidirectional communication.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

try:
	import socketio
	from socketio import AsyncServer, AsyncClient
	from aiohttp import web, WSMsgType
	import aiohttp_cors
except ImportError:
	print("Socket.IO dependencies not installed. Run: pip install python-socketio aiohttp aiohttp-cors")
	socketio = None
	AsyncServer = None
	AsyncClient = None
	web = None

logger = logging.getLogger(__name__)


class SocketIONamespace(Enum):
	"""Socket.IO namespaces for different functionality"""
	COLLABORATION = "/collaboration"
	PRESENCE = "/presence"
	FILE_TRANSFER = "/file-transfer"
	VOICE_CHAT = "/voice-chat"
	NOTIFICATIONS = "/notifications"
	ANALYTICS = "/analytics"
	ADMIN = "/admin"


class SocketIOEventType(Enum):
	"""Socket.IO event types"""
	CONNECT = "connect"
	DISCONNECT = "disconnect"
	MESSAGE = "message"
	COLLABORATION_EVENT = "collaboration_event"
	PRESENCE_UPDATE = "presence_update"
	CURSOR_MOVE = "cursor_move"
	TEXT_EDIT = "text_edit"
	FILE_SHARE = "file_share"
	VOICE_REQUEST = "voice_request"
	NOTIFICATION = "notification"
	HEARTBEAT = "heartbeat"


@dataclass
class SocketIOMessage:
	"""Socket.IO message structure"""
	event_type: SocketIOEventType
	namespace: SocketIONamespace
	data: Dict[str, Any]
	sender_id: str
	room: Optional[str] = None
	timestamp: datetime = field(default_factory=datetime.utcnow)
	message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"event_type": self.event_type.value,
			"namespace": self.namespace.value,
			"data": self.data,
			"sender_id": self.sender_id,
			"room": self.room,
			"timestamp": self.timestamp.isoformat(),
			"message_id": self.message_id
		}


@dataclass
class SocketIOClient:
	"""Socket.IO client information"""
	session_id: str
	user_id: str
	namespaces: Set[str] = field(default_factory=set)
	rooms: Set[str] = field(default_factory=set)
	connected_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	transport: str = "websocket"
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"session_id": self.session_id,
			"user_id": self.user_id,
			"namespaces": list(self.namespaces),
			"rooms": list(self.rooms),
			"connected_at": self.connected_at.isoformat(),
			"last_activity": self.last_activity.isoformat(),
			"ip_address": self.ip_address,
			"user_agent": self.user_agent,
			"transport": self.transport
		}


class SocketIOProtocolManager:
	"""Manages Socket.IO protocol for real-time collaboration"""
	
	def __init__(self, host: str = "localhost", port: int = 3000):
		self.host = host
		self.port = port
		
		# Socket.IO server
		self.sio: Optional[AsyncServer] = None
		self.app: Optional[web.Application] = None
		self.runner: Optional[web.AppRunner] = None
		self.site: Optional[web.TCPSite] = None
		
		# Client management
		self.clients: Dict[str, SocketIOClient] = {}
		self.rooms: Dict[str, Set[str]] = {}  # room_id -> set of session_ids
		self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
		
		# Event handlers
		self.event_handlers: Dict[str, List[Callable]] = {}
		self.namespace_handlers: Dict[str, Dict[str, Callable]] = {}
		
		# Message history
		self.message_history: List[SocketIOMessage] = []
		self.max_history_size = 1000
		
		# Configuration
		self.cors_allowed_origins = ["*"]
		self.max_http_buffer_size = 1000000  # 1MB
		
		# Statistics
		self.stats = {
			"connections": 0,
			"disconnections": 0,
			"messages_sent": 0,
			"messages_received": 0,
			"bytes_sent": 0,
			"bytes_received": 0,
			"rooms_created": 0,
			"uptime_start": None
		}
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize Socket.IO server"""
		try:
			if not AsyncServer:
				return {"error": "Socket.IO dependencies not installed"}
			
			# Create Socket.IO server
			self.sio = AsyncServer(
				cors_allowed_origins=self.cors_allowed_origins,
				max_http_buffer_size=self.max_http_buffer_size,
				async_mode='aiohttp',
				ping_timeout=60,
				ping_interval=25
			)
			
			# Create aiohttp application
			self.app = web.Application()
			self.sio.attach(self.app)
			
			# Configure CORS
			cors = aiohttp_cors.setup(self.app, defaults={
				"*": aiohttp_cors.ResourceOptions(
					allow_credentials=True,
					expose_headers="*",
					allow_headers="*",
					allow_methods="*"
				)
			})
			
			# Add routes
			self.app.router.add_get('/', self._index_handler)
			self.app.router.add_get('/status', self._status_handler)
			
			# Apply CORS to all routes
			for route in list(self.app.router.routes()):
				cors.add(route)
			
			# Register Socket.IO event handlers
			await self._register_socketio_handlers()
			
			# Start server
			self.runner = web.AppRunner(self.app)
			await self.runner.setup()
			
			self.site = web.TCPSite(self.runner, self.host, self.port)
			await self.site.start()
			
			self.stats["uptime_start"] = datetime.utcnow()
			
			logger.info(f"Socket.IO server started on http://{self.host}:{self.port}")
			
			return {
				"status": "started",
				"address": f"http://{self.host}:{self.port}",
				"namespaces": [ns.value for ns in SocketIONamespace]
			}
			
		except Exception as e:
			logger.error(f"Failed to initialize Socket.IO server: {e}")
			return {"error": f"Socket.IO initialization failed: {str(e)}"}
	
	async def shutdown(self) -> Dict[str, Any]:
		"""Shutdown Socket.IO server"""
		try:
			if self.site:
				await self.site.stop()
			
			if self.runner:
				await self.runner.cleanup()
			
			logger.info("Socket.IO server stopped")
			
			return {"status": "stopped"}
			
		except Exception as e:
			logger.error(f"Error stopping Socket.IO server: {e}")
			return {"error": f"Shutdown failed: {str(e)}"}
	
	async def _register_socketio_handlers(self):
		"""Register Socket.IO event handlers"""
		
		@self.sio.event
		async def connect(sid, environ, auth):
			"""Handle client connection"""
			try:
				# Extract user information
				user_id = auth.get('user_id', f'anonymous_{sid[:8]}') if auth else f'anonymous_{sid[:8]}'
				
				# Create client record
				client = SocketIOClient(
					session_id=sid,
					user_id=user_id,
					ip_address=environ.get('REMOTE_ADDR'),
					user_agent=environ.get('HTTP_USER_AGENT')
				)
				
				self.clients[sid] = client
				
				# Track user sessions
				if user_id not in self.user_sessions:
					self.user_sessions[user_id] = set()
				self.user_sessions[user_id].add(sid)
				
				# Update statistics
				self.stats["connections"] += 1
				
				logger.info(f"Socket.IO client connected: {sid} (user: {user_id})")
				
				# Send welcome message
				await self.sio.emit('connected', {
					'session_id': sid,
					'user_id': user_id,
					'server_time': datetime.utcnow().isoformat()
				}, room=sid)
				
				# Broadcast presence update
				await self._broadcast_presence_update(user_id, 'online')
				
				return True
				
			except Exception as e:
				logger.error(f"Error in Socket.IO connect handler: {e}")
				return False
		
		@self.sio.event
		async def disconnect(sid):
			"""Handle client disconnection"""
			try:
				if sid in self.clients:
					client = self.clients[sid]
					user_id = client.user_id
					
					# Remove from rooms
					for room in list(client.rooms):
						await self.leave_room(sid, room)
					
					# Remove from user sessions
					if user_id in self.user_sessions:
						self.user_sessions[user_id].discard(sid)
						if not self.user_sessions[user_id]:
							del self.user_sessions[user_id]
							# User fully disconnected
							await self._broadcast_presence_update(user_id, 'offline')
					
					# Remove client record
					del self.clients[sid]
					
					# Update statistics
					self.stats["disconnections"] += 1
					
					logger.info(f"Socket.IO client disconnected: {sid} (user: {user_id})")
				
			except Exception as e:
				logger.error(f"Error in Socket.IO disconnect handler: {e}")
		
		@self.sio.event
		async def join_room(sid, data):
			"""Handle room join request"""
			try:
				room_id = data.get('room_id')
				if not room_id:
					await self.sio.emit('error', {'message': 'Room ID required'}, room=sid)
					return
				
				result = await self.join_room(sid, room_id)
				await self.sio.emit('room_joined', result, room=sid)
				
			except Exception as e:
				logger.error(f"Error in join_room handler: {e}")
				await self.sio.emit('error', {'message': str(e)}, room=sid)
		
		@self.sio.event
		async def leave_room(sid, data):
			"""Handle room leave request"""
			try:
				room_id = data.get('room_id')
				if not room_id:
					await self.sio.emit('error', {'message': 'Room ID required'}, room=sid)
					return
				
				result = await self.leave_room(sid, room_id)
				await self.sio.emit('room_left', result, room=sid)
				
			except Exception as e:
				logger.error(f"Error in leave_room handler: {e}")
				await self.sio.emit('error', {'message': str(e)}, room=sid)
		
		@self.sio.event
		async def collaboration_event(sid, data):
			"""Handle collaboration event"""
			try:
				await self._handle_collaboration_event(sid, data)
				
			except Exception as e:
				logger.error(f"Error in collaboration_event handler: {e}")
				await self.sio.emit('error', {'message': str(e)}, room=sid)
		
		@self.sio.event
		async def cursor_move(sid, data):
			"""Handle cursor movement"""
			try:
				await self._handle_cursor_move(sid, data)
				
			except Exception as e:
				logger.error(f"Error in cursor_move handler: {e}")
		
		@self.sio.event
		async def text_edit(sid, data):
			"""Handle text editing"""
			try:
				await self._handle_text_edit(sid, data)
				
			except Exception as e:
				logger.error(f"Error in text_edit handler: {e}")
		
		@self.sio.event
		async def file_share(sid, data):
			"""Handle file sharing"""
			try:
				await self._handle_file_share(sid, data)
				
			except Exception as e:
				logger.error(f"Error in file_share handler: {e}")
		
		@self.sio.event
		async def heartbeat(sid, data):
			"""Handle heartbeat"""
			try:
				if sid in self.clients:
					self.clients[sid].last_activity = datetime.utcnow()
				
				await self.sio.emit('heartbeat_ack', {
					'timestamp': datetime.utcnow().isoformat()
				}, room=sid)
				
			except Exception as e:
				logger.error(f"Error in heartbeat handler: {e}")
	
	async def join_room(self, session_id: str, room_id: str) -> Dict[str, Any]:
		"""Add client to room"""
		try:
			if session_id not in self.clients:
				return {"error": "Client not found"}
			
			client = self.clients[session_id]
			
			# Add to Socket.IO room
			await self.sio.enter_room(session_id, room_id)
			
			# Update client record
			client.rooms.add(room_id)
			
			# Update room tracking
			if room_id not in self.rooms:
				self.rooms[room_id] = set()
				self.stats["rooms_created"] += 1
			self.rooms[room_id].add(session_id)
			
			logger.info(f"Client {session_id} joined room {room_id}")
			
			# Notify other room members
			await self.sio.emit('user_joined', {
				'user_id': client.user_id,
				'room_id': room_id,
				'timestamp': datetime.utcnow().isoformat()
			}, room=room_id, skip_sid=session_id)
			
			return {
				"status": "joined",
				"room_id": room_id,
				"member_count": len(self.rooms[room_id])
			}
			
		except Exception as e:
			logger.error(f"Error joining room: {e}")
			return {"error": f"Failed to join room: {str(e)}"}
	
	async def leave_room(self, session_id: str, room_id: str) -> Dict[str, Any]:
		"""Remove client from room"""
		try:
			if session_id not in self.clients:
				return {"error": "Client not found"}
			
			client = self.clients[session_id]
			
			# Remove from Socket.IO room
			await self.sio.leave_room(session_id, room_id)
			
			# Update client record
			client.rooms.discard(room_id)
			
			# Update room tracking
			if room_id in self.rooms:
				self.rooms[room_id].discard(session_id)
				if not self.rooms[room_id]:
					del self.rooms[room_id]
			
			logger.info(f"Client {session_id} left room {room_id}")
			
			# Notify other room members
			if room_id in self.rooms:
				await self.sio.emit('user_left', {
					'user_id': client.user_id,
					'room_id': room_id,
					'timestamp': datetime.utcnow().isoformat()
				}, room=room_id)
			
			return {
				"status": "left",
				"room_id": room_id
			}
			
		except Exception as e:
			logger.error(f"Error leaving room: {e}")
			return {"error": f"Failed to leave room: {str(e)}"}
	
	async def emit_to_room(self, room_id: str, event: str, data: Any) -> Dict[str, Any]:
		"""Emit event to all clients in room"""
		try:
			await self.sio.emit(event, data, room=room_id)
			
			# Update statistics
			member_count = len(self.rooms.get(room_id, set()))
			self.stats["messages_sent"] += member_count
			
			return {
				"status": "sent",
				"room_id": room_id,
				"member_count": member_count
			}
			
		except Exception as e:
			logger.error(f"Error emitting to room: {e}")
			return {"error": f"Failed to emit to room: {str(e)}"}
	
	async def emit_to_user(self, user_id: str, event: str, data: Any) -> Dict[str, Any]:
		"""Emit event to all sessions of a user"""
		try:
			if user_id not in self.user_sessions:
				return {"error": "User not connected"}
			
			sessions_sent = 0
			for session_id in self.user_sessions[user_id]:
				await self.sio.emit(event, data, room=session_id)
				sessions_sent += 1
			
			# Update statistics
			self.stats["messages_sent"] += sessions_sent
			
			return {
				"status": "sent",
				"user_id": user_id,
				"sessions_sent": sessions_sent
			}
			
		except Exception as e:
			logger.error(f"Error emitting to user: {e}")
			return {"error": f"Failed to emit to user: {str(e)}"}
	
	async def broadcast(self, event: str, data: Any, namespace: str = None) -> Dict[str, Any]:
		"""Broadcast event to all connected clients"""
		try:
			await self.sio.emit(event, data, namespace=namespace)
			
			# Update statistics
			self.stats["messages_sent"] += len(self.clients)
			
			return {
				"status": "broadcast",
				"client_count": len(self.clients)
			}
			
		except Exception as e:
			logger.error(f"Error broadcasting: {e}")
			return {"error": f"Failed to broadcast: {str(e)}"}
	
	async def _handle_collaboration_event(self, session_id: str, data: Dict[str, Any]):
		"""Handle collaboration event from client"""
		if session_id not in self.clients:
			return
		
		client = self.clients[session_id]
		event_type = data.get('event_type')
		
		# Create message
		message = SocketIOMessage(
			event_type=SocketIOEventType.COLLABORATION_EVENT,
			namespace=SocketIONamespace.COLLABORATION,
			data=data,
			sender_id=client.user_id
		)
		
		# Add to history
		self._add_to_history(message)
		
		# Broadcast to rooms the client is in
		for room_id in client.rooms:
			await self.sio.emit('collaboration_event', {
				**data,
				'sender_id': client.user_id,
				'timestamp': message.timestamp.isoformat()
			}, room=room_id, skip_sid=session_id)
		
		# Update statistics
		self.stats["messages_received"] += 1
		client.last_activity = datetime.utcnow()
	
	async def _handle_cursor_move(self, session_id: str, data: Dict[str, Any]):
		"""Handle cursor movement from client"""
		if session_id not in self.clients:
			return
		
		client = self.clients[session_id]
		
		# Broadcast cursor position to rooms
		for room_id in client.rooms:
			await self.sio.emit('cursor_move', {
				**data,
				'user_id': client.user_id,
				'timestamp': datetime.utcnow().isoformat()
			}, room=room_id, skip_sid=session_id)
		
		client.last_activity = datetime.utcnow()
	
	async def _handle_text_edit(self, session_id: str, data: Dict[str, Any]):
		"""Handle text editing from client"""
		if session_id not in self.clients:
			return
		
		client = self.clients[session_id]
		
		# Broadcast text edit to rooms
		for room_id in client.rooms:
			await self.sio.emit('text_edit', {
				**data,
				'user_id': client.user_id,
				'timestamp': datetime.utcnow().isoformat()
			}, room=room_id, skip_sid=session_id)
		
		client.last_activity = datetime.utcnow()
	
	async def _handle_file_share(self, session_id: str, data: Dict[str, Any]):
		"""Handle file sharing from client"""
		if session_id not in self.clients:
			return
		
		client = self.clients[session_id]
		
		# Broadcast file share to rooms
		for room_id in client.rooms:
			await self.sio.emit('file_share', {
				**data,
				'sender_id': client.user_id,
				'timestamp': datetime.utcnow().isoformat()
			}, room=room_id, skip_sid=session_id)
		
		client.last_activity = datetime.utcnow()
	
	async def _broadcast_presence_update(self, user_id: str, status: str):
		"""Broadcast presence update to all clients"""
		await self.broadcast('presence_update', {
			'user_id': user_id,
			'status': status,
			'timestamp': datetime.utcnow().isoformat()
		})
	
	def _add_to_history(self, message: SocketIOMessage):
		"""Add message to history with size management"""
		self.message_history.append(message)
		
		# Maintain history size limit
		if len(self.message_history) > self.max_history_size:
			self.message_history = self.message_history[-self.max_history_size:]
	
	async def _index_handler(self, request):
		"""Handle index page request"""
		return web.Response(text="""
<!DOCTYPE html>
<html>
<head>
    <title>APG Real-Time Collaboration - Socket.IO</title>
</head>
<body>
    <h1>APG Real-Time Collaboration Socket.IO Server</h1>
    <p>Server is running and ready for connections.</p>
    <p>Status: <a href="/status">Check Status</a></p>
</body>
</html>
		""", content_type='text/html')
	
	async def _status_handler(self, request):
		"""Handle status request"""
		stats = self.get_statistics()
		return web.json_response(stats)
	
	def get_connected_clients(self) -> List[Dict[str, Any]]:
		"""Get list of connected clients"""
		return [client.to_dict() for client in self.clients.values()]
	
	def get_rooms(self) -> Dict[str, Dict[str, Any]]:
		"""Get information about active rooms"""
		return {
			room_id: {
				"member_count": len(members),
				"members": [
					self.clients[sid].user_id for sid in members 
					if sid in self.clients
				]
			}
			for room_id, members in self.rooms.items()
		}
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get Socket.IO protocol statistics"""
		uptime_seconds = 0
		if self.stats["uptime_start"]:
			uptime_seconds = int((datetime.utcnow() - self.stats["uptime_start"]).total_seconds())
		
		return {
			**self.stats,
			"address": f"http://{self.host}:{self.port}",
			"connected_clients": len(self.clients),
			"active_rooms": len(self.rooms),
			"unique_users": len(self.user_sessions),
			"message_history_size": len(self.message_history),
			"uptime_seconds": uptime_seconds,
			"uptime_start": self.stats["uptime_start"].isoformat() if self.stats["uptime_start"] else None
		}


# Global Socket.IO manager instance
socketio_protocol_manager = None


async def initialize_socketio_protocol(host: str = "localhost", port: int = 3000) -> Dict[str, Any]:
	"""Initialize global Socket.IO protocol manager"""
	global socketio_protocol_manager
	
	socketio_protocol_manager = SocketIOProtocolManager(host=host, port=port)
	
	result = await socketio_protocol_manager.initialize()
	
	return result


def get_socketio_manager() -> Optional[SocketIOProtocolManager]:
	"""Get global Socket.IO protocol manager"""
	return socketio_protocol_manager


if __name__ == "__main__":
	# Test Socket.IO protocol implementation
	async def test_socketio():
		print("Testing Socket.IO protocol implementation...")
		
		# Initialize Socket.IO manager
		result = await initialize_socketio_protocol()
		print(f"Socket.IO initialization result: {result}")
		
		if result.get("status") == "started":
			manager = get_socketio_manager()
			
			# Simulate some activity
			await asyncio.sleep(1)
			
			# Get statistics
			stats = manager.get_statistics()
			print(f"Socket.IO statistics: {stats}")
			
			# Test broadcast
			broadcast_result = await manager.broadcast('test_event', {
				'message': 'Test broadcast message',
				'timestamp': datetime.utcnow().isoformat()
			})
			print(f"Broadcast result: {broadcast_result}")
			
			# Keep server running for a bit
			print("Socket.IO server running. Connect with a client to test...")
			await asyncio.sleep(5)
			
			# Shutdown
			shutdown_result = await manager.shutdown()
			print(f"Shutdown result: {shutdown_result}")
		
		print("✅ Socket.IO protocol test completed")
	
	asyncio.run(test_socketio())
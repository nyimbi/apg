"""
APG NLP WebSocket Streaming Integration

Real-time WebSocket-based streaming NLP processing with bi-directional communication,
session management, and automatic reconnection support.

Features:
- WebSocket server for real-time NLP streaming
- Session-based processing with state management
- Automatic chunk processing and result streaming
- Error handling and recovery mechanisms
- Performance monitoring and throttling
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import uuid
from contextlib import asynccontextmanager
from uuid_extensions import uuid7str

try:
	import websockets
	from websockets.server import WebSocketServerProtocol
	WEBSOCKETS_AVAILABLE = True
except ImportError:
	WEBSOCKETS_AVAILABLE = False
	WebSocketServerProtocol = None

from models import (
	ProcessingRequest, ProcessingResult, StreamingSession, StreamingChunk,
	NLPTaskType, ModelProvider, QualityLevel, LanguageCode, ProcessingStatus
)
from processing_pipeline import AdvancedProcessingPipeline, StreamingProcessor

# Configure logging
logger = logging.getLogger(__name__)

class MessageType(str, Enum):
	"""WebSocket message types"""
	CONNECT = "connect"
	DISCONNECT = "disconnect"
	START_SESSION = "start_session"
	END_SESSION = "end_session"
	TEXT_CHUNK = "text_chunk"
	PROCESSING_RESULT = "processing_result"
	ERROR = "error"
	HEARTBEAT = "heartbeat"
	STATUS_UPDATE = "status_update"

class ConnectionState(str, Enum):
	"""WebSocket connection states"""
	CONNECTING = "connecting"
	CONNECTED = "connected"
	PROCESSING = "processing"
	IDLE = "idle"
	DISCONNECTING = "disconnecting"
	DISCONNECTED = "disconnected"
	ERROR = "error"

@dataclass
class WebSocketMessage:
	"""WebSocket message structure"""
	type: MessageType
	session_id: Optional[str] = None
	data: Dict[str, Any] = field(default_factory=dict)
	timestamp: datetime = field(default_factory=datetime.utcnow)
	message_id: str = field(default_factory=uuid7str)

@dataclass
class WebSocketConnection:
	"""WebSocket connection information"""
	connection_id: str
	websocket: Optional[Any] = None  # WebSocketServerProtocol
	tenant_id: Optional[str] = None
	user_id: Optional[str] = None
	session_id: Optional[str] = None
	state: ConnectionState = ConnectionState.CONNECTING
	connected_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	message_count: int = 0
	error_count: int = 0
	metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamingStats:
	"""Streaming statistics"""
	total_connections: int = 0
	active_connections: int = 0
	active_sessions: int = 0
	total_messages: int = 0
	total_chunks_processed: int = 0
	average_latency_ms: float = 0.0
	error_rate: float = 0.0
	uptime_seconds: float = 0.0

class WebSocketStreamingManager:
	"""WebSocket streaming manager for real-time NLP processing"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for WebSocket streaming manager"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Connection management
		self.active_connections: Dict[str, WebSocketConnection] = {}
		self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
		self.tenant_connections: Dict[str, Set[str]] = defaultdict(set)
		
		# Processing components
		self.pipeline = AdvancedProcessingPipeline(tenant_id, self.config.get("pipeline", {}))
		self.streaming_processor = self.pipeline.streaming_processor
		
		# Performance tracking
		self.message_stats: Dict[MessageType, int] = defaultdict(int)
		self.latency_history: deque = deque(maxlen=1000)
		self.error_history: deque = deque(maxlen=100)
		self.start_time = datetime.utcnow()
		
		# Configuration
		self._setup_streaming_config()
		self._log_manager_initialized()
	
	def _setup_streaming_config(self) -> None:
		"""Setup streaming configuration"""
		self.max_connections = self.config.get("max_connections", 1000)
		self.max_sessions_per_connection = self.config.get("max_sessions_per_connection", 5)
		self.heartbeat_interval = self.config.get("heartbeat_interval", 30)
		self.chunk_timeout = self.config.get("chunk_timeout", 30)
		self.max_message_size = self.config.get("max_message_size", 64 * 1024)  # 64KB
		self.rate_limit_messages = self.config.get("rate_limit_messages", 100)  # per minute
		self.rate_limit_window = self.config.get("rate_limit_window", 60)  # seconds
	
	def _log_manager_initialized(self) -> None:
		"""Log manager initialization"""
		logger.info(f"WebSocket streaming manager initialized for tenant: {self.tenant_id}")
		if not WEBSOCKETS_AVAILABLE:
			logger.warning("WebSockets library not available - WebSocket functionality disabled")
	
	async def handle_connection(self, websocket: Any, path: str) -> None:
		"""Handle new WebSocket connection"""
		if not WEBSOCKETS_AVAILABLE:
			logger.error("WebSocket connection attempted but websockets library not available")
			return
		
		connection_id = uuid7str()
		connection = WebSocketConnection(
			connection_id=connection_id,
			websocket=websocket
		)
		
		self.active_connections[connection_id] = connection
		
		try:
			logger.info(f"WebSocket connection established: {connection_id}")
			await self._connection_handler(connection)
		
		except websockets.exceptions.ConnectionClosed:
			logger.info(f"WebSocket connection closed: {connection_id}")
		
		except Exception as e:
			logger.error(f"WebSocket connection error ({connection_id}): {e}")
			connection.state = ConnectionState.ERROR
			connection.error_count += 1
		
		finally:
			await self._cleanup_connection(connection_id)
	
	async def _connection_handler(self, connection: WebSocketConnection) -> None:
		"""Handle WebSocket connection lifecycle"""
		try:
			# Send welcome message
			welcome_msg = WebSocketMessage(
				type=MessageType.CONNECT,
				data={
					"connection_id": connection.connection_id,
					"server_time": datetime.utcnow().isoformat(),
					"capabilities": ["nlp_streaming", "real_time_processing"]
				}
			)
			await self._send_message(connection, welcome_msg)
			
			connection.state = ConnectionState.CONNECTED
			
			# Start heartbeat task
			heartbeat_task = asyncio.create_task(
				self._heartbeat_handler(connection)
			)
			
			try:
				async for raw_message in connection.websocket:
					await self._process_message(connection, raw_message)
			
			finally:
				heartbeat_task.cancel()
				try:
					await heartbeat_task
				except asyncio.CancelledError:
					pass
		
		except Exception as e:
			logger.error(f"Connection handler error: {e}")
			raise
	
	async def _process_message(self, connection: WebSocketConnection, raw_message: str) -> None:
		"""Process incoming WebSocket message"""
		try:
			# Rate limiting check
			if not self._check_rate_limit(connection):
				error_msg = WebSocketMessage(
					type=MessageType.ERROR,
					data={"error": "Rate limit exceeded", "code": "RATE_LIMIT"}
				)
				await self._send_message(connection, error_msg)
				return
			
			# Parse message
			if len(raw_message) > self.max_message_size:
				error_msg = WebSocketMessage(
					type=MessageType.ERROR,
					data={"error": "Message too large", "code": "MESSAGE_SIZE"}
				)
				await self._send_message(connection, error_msg)
				return
			
			try:
				message_data = json.loads(raw_message)
			except json.JSONDecodeError as e:
				error_msg = WebSocketMessage(
					type=MessageType.ERROR,
					data={"error": f"Invalid JSON: {str(e)}", "code": "INVALID_JSON"}
				)
				await self._send_message(connection, error_msg)
				return
			
			# Create message object
			message = WebSocketMessage(
				type=MessageType(message_data.get("type", "unknown")),
				session_id=message_data.get("session_id"),
				data=message_data.get("data", {})
			)
			
			# Update connection activity
			connection.last_activity = datetime.utcnow()
			connection.message_count += 1
			self.message_stats[message.type] += 1
			
			# Route message to appropriate handler
			await self._route_message(connection, message)
		
		except Exception as e:
			logger.error(f"Message processing error: {e}")
			self.error_history.append({
				"timestamp": datetime.utcnow(),
				"connection_id": connection.connection_id,
				"error": str(e),
				"type": "message_processing"
			})
			
			error_msg = WebSocketMessage(
				type=MessageType.ERROR,
				data={"error": "Message processing failed", "code": "PROCESSING_ERROR"}
			)
			await self._send_message(connection, error_msg)
	
	def _check_rate_limit(self, connection: WebSocketConnection) -> bool:
		"""Check if connection is within rate limits"""
		now = datetime.utcnow()
		window_start = now - timedelta(seconds=self.rate_limit_window)
		
		# Simple rate limiting based on message count
		recent_messages = getattr(connection, 'recent_message_times', deque(maxlen=self.rate_limit_messages))
		
		# Clean old messages
		while recent_messages and recent_messages[0] < window_start:
			recent_messages.popleft()
		
		# Check limit
		if len(recent_messages) >= self.rate_limit_messages:
			return False
		
		# Add current message
		recent_messages.append(now)
		connection.recent_message_times = recent_messages
		
		return True
	
	async def _route_message(self, connection: WebSocketConnection, message: WebSocketMessage) -> None:
		"""Route message to appropriate handler"""
		if message.type == MessageType.START_SESSION:
			await self._handle_start_session(connection, message)
		
		elif message.type == MessageType.END_SESSION:
			await self._handle_end_session(connection, message)
		
		elif message.type == MessageType.TEXT_CHUNK:
			await self._handle_text_chunk(connection, message)
		
		elif message.type == MessageType.HEARTBEAT:
			await self._handle_heartbeat(connection, message)
		
		else:
			error_msg = WebSocketMessage(
				type=MessageType.ERROR,
				data={"error": f"Unknown message type: {message.type}", "code": "UNKNOWN_TYPE"}
			)
			await self._send_message(connection, error_msg)
	
	async def _handle_start_session(self, connection: WebSocketConnection, message: WebSocketMessage) -> None:
		"""Handle session start request"""
		try:
			# Extract session parameters
			data = message.data
			task_type = NLPTaskType(data.get("task_type", "sentiment_analysis"))
			user_id = data.get("user_id", connection.connection_id)
			config = data.get("config", {})
			
			# Set connection tenant and user
			connection.tenant_id = data.get("tenant_id", self.tenant_id)
			connection.user_id = user_id
			
			# Validate tenant access
			if connection.tenant_id != self.tenant_id:
				error_msg = WebSocketMessage(
					type=MessageType.ERROR,
					data={"error": "Tenant access denied", "code": "TENANT_ACCESS"}
				)
				await self._send_message(connection, error_msg)
				return
			
			# Create streaming session
			session = await self.streaming_processor.create_session(
				user_id=user_id,
				task_type=task_type,
				config=config
			)
			
			# Associate session with connection
			connection.session_id = session.id
			self.session_connections[session.id] = connection.connection_id
			self.tenant_connections[connection.tenant_id].add(connection.connection_id)
			
			connection.state = ConnectionState.PROCESSING
			
			# Start result streaming task
			asyncio.create_task(self._stream_results(connection, session))
			
			# Send confirmation
			response = WebSocketMessage(
				type=MessageType.START_SESSION,
				session_id=session.id,
				data={
					"session_id": session.id,
					"task_type": task_type.value,
					"status": "started",
					"config": {
						"chunk_size": session.chunk_size,
						"overlap_size": session.overlap_size
					}
				}
			)
			await self._send_message(connection, response)
			
			logger.info(f"Streaming session started: {session.id} for connection: {connection.connection_id}")
		
		except Exception as e:
			logger.error(f"Session start error: {e}")
			error_msg = WebSocketMessage(
				type=MessageType.ERROR,
				data={"error": f"Failed to start session: {str(e)}", "code": "SESSION_START"}
			)
			await self._send_message(connection, error_msg)
	
	async def _handle_end_session(self, connection: WebSocketConnection, message: WebSocketMessage) -> None:
		"""Handle session end request"""
		try:
			session_id = message.session_id or connection.session_id
			
			if not session_id:
				error_msg = WebSocketMessage(
					type=MessageType.ERROR,
					data={"error": "No active session", "code": "NO_SESSION"}
				)
				await self._send_message(connection, error_msg)
				return
			
			# Close streaming session
			success = await self.streaming_processor.close_session(session_id)
			
			# Update connection state
			connection.session_id = None
			connection.state = ConnectionState.IDLE
			
			# Cleanup associations
			if session_id in self.session_connections:
				del self.session_connections[session_id]
			
			# Send confirmation
			response = WebSocketMessage(
				type=MessageType.END_SESSION,
				session_id=session_id,
				data={
					"session_id": session_id,
					"status": "ended" if success else "error"
				}
			)
			await self._send_message(connection, response)
			
			logger.info(f"Streaming session ended: {session_id}")
		
		except Exception as e:
			logger.error(f"Session end error: {e}")
			error_msg = WebSocketMessage(
				type=MessageType.ERROR,
				data={"error": f"Failed to end session: {str(e)}", "code": "SESSION_END"}
			)
			await self._send_message(connection, error_msg)
	
	async def _handle_text_chunk(self, connection: WebSocketConnection, message: WebSocketMessage) -> None:
		"""Handle incoming text chunk"""
		try:
			session_id = message.session_id or connection.session_id
			
			if not session_id:
				error_msg = WebSocketMessage(
					type=MessageType.ERROR,
					data={"error": "No active session", "code": "NO_SESSION"}
				)
				await self._send_message(connection, error_msg)
				return
			
			text_content = message.data.get("text", "")
			
			if not text_content:
				error_msg = WebSocketMessage(
					type=MessageType.ERROR,
					data={"error": "Empty text chunk", "code": "EMPTY_TEXT"}
				)
				await self._send_message(connection, error_msg)
				return
			
			# Add chunk to streaming processor
			success = await self.streaming_processor.add_chunk(session_id, text_content)
			
			if not success:
				error_msg = WebSocketMessage(
					type=MessageType.ERROR,
					data={"error": "Failed to add chunk", "code": "CHUNK_ADD"}
				)
				await self._send_message(connection, error_msg)
				return
			
			# Send acknowledgment
			ack_msg = WebSocketMessage(
				type=MessageType.STATUS_UPDATE,
				session_id=session_id,
				data={
					"status": "chunk_received",
					"chunk_size": len(text_content),
					"timestamp": datetime.utcnow().isoformat()
				}
			)
			await self._send_message(connection, ack_msg)
		
		except Exception as e:
			logger.error(f"Text chunk handling error: {e}")
			error_msg = WebSocketMessage(
				type=MessageType.ERROR,
				data={"error": f"Chunk processing failed: {str(e)}", "code": "CHUNK_ERROR"}
			)
			await self._send_message(connection, error_msg)
	
	async def _handle_heartbeat(self, connection: WebSocketConnection, message: WebSocketMessage) -> None:
		"""Handle heartbeat message"""
		connection.last_activity = datetime.utcnow()
		
		response = WebSocketMessage(
			type=MessageType.HEARTBEAT,
			data={
				"server_time": datetime.utcnow().isoformat(),
				"connection_id": connection.connection_id
			}
		)
		await self._send_message(connection, response)
	
	async def _stream_results(self, connection: WebSocketConnection, session: StreamingSession) -> None:
		"""Stream processing results to client"""
		try:
			while connection.state == ConnectionState.PROCESSING and session.status == "active":
				# Get next result
				result = await self.streaming_processor.get_result(session.id, timeout=5.0)
				
				if result:
					# Track latency
					latency_ms = result.total_time_ms
					self.latency_history.append(latency_ms)
					
					# Send result
					result_msg = WebSocketMessage(
						type=MessageType.PROCESSING_RESULT,
						session_id=session.id,
						data={
							"result_id": result.id,
							"task_type": result.task_type.value,
							"results": result.results,
							"confidence": result.confidence_score,
							"processing_time_ms": result.processing_time_ms,
							"status": result.status
						}
					)
					await self._send_message(connection, result_msg)
		
		except asyncio.CancelledError:
			logger.info(f"Result streaming cancelled for session: {session.id}")
		
		except Exception as e:
			logger.error(f"Result streaming error: {e}")
			error_msg = WebSocketMessage(
				type=MessageType.ERROR,
				data={"error": f"Result streaming failed: {str(e)}", "code": "STREAMING_ERROR"}
			)
			await self._send_message(connection, error_msg)
	
	async def _heartbeat_handler(self, connection: WebSocketConnection) -> None:
		"""Handle connection heartbeat"""
		try:
			while connection.state in [ConnectionState.CONNECTED, ConnectionState.PROCESSING, ConnectionState.IDLE]:
				await asyncio.sleep(self.heartbeat_interval)
				
				# Check if connection is stale
				if datetime.utcnow() - connection.last_activity > timedelta(seconds=self.heartbeat_interval * 3):
					logger.warning(f"Connection stale, closing: {connection.connection_id}")
					break
				
				# Send heartbeat
				heartbeat_msg = WebSocketMessage(
					type=MessageType.HEARTBEAT,
					data={"server_time": datetime.utcnow().isoformat()}
				)
				await self._send_message(connection, heartbeat_msg)
		
		except asyncio.CancelledError:
			logger.info(f"Heartbeat handler cancelled for: {connection.connection_id}")
		
		except Exception as e:
			logger.error(f"Heartbeat error: {e}")
	
	async def _send_message(self, connection: WebSocketConnection, message: WebSocketMessage) -> bool:
		"""Send message to WebSocket client"""
		if not WEBSOCKETS_AVAILABLE:
			return False
		
		try:
			message_data = {
				"type": message.type.value,
				"message_id": message.message_id,
				"timestamp": message.timestamp.isoformat(),
				"session_id": message.session_id,
				"data": message.data
			}
			
			await connection.websocket.send(json.dumps(message_data))
			return True
		
		except websockets.exceptions.ConnectionClosed:
			logger.info(f"Connection closed while sending message: {connection.connection_id}")
			return False
		
		except Exception as e:
			logger.error(f"Message send error: {e}")
			connection.error_count += 1
			return False
	
	async def _cleanup_connection(self, connection_id: str) -> None:
		"""Cleanup connection resources"""
		if connection_id not in self.active_connections:
			return
		
		connection = self.active_connections[connection_id]
		
		# Close associated session
		if connection.session_id:
			await self.streaming_processor.close_session(connection.session_id)
			if connection.session_id in self.session_connections:
				del self.session_connections[connection.session_id]
		
		# Remove from tenant connections
		if connection.tenant_id:
			self.tenant_connections[connection.tenant_id].discard(connection_id)
		
		# Remove connection
		del self.active_connections[connection_id]
		
		logger.info(f"Connection cleanup completed: {connection_id}")
	
	def get_streaming_stats(self) -> StreamingStats:
		"""Get comprehensive streaming statistics"""
		now = datetime.utcnow()
		uptime = (now - self.start_time).total_seconds()
		
		# Calculate averages
		if self.latency_history:
			avg_latency = sum(self.latency_history) / len(self.latency_history)
		else:
			avg_latency = 0.0
		
		total_messages = sum(self.message_stats.values())
		error_rate = len(self.error_history) / max(total_messages, 1) * 100
		
		return StreamingStats(
			total_connections=len(self.active_connections),
			active_connections=len([c for c in self.active_connections.values() 
								   if c.state in [ConnectionState.CONNECTED, ConnectionState.PROCESSING]]),
			active_sessions=len(self.streaming_processor.active_sessions),
			total_messages=total_messages,
			total_chunks_processed=sum(session.chunks_processed 
									  for session in self.streaming_processor.active_sessions.values()),
			average_latency_ms=round(avg_latency, 2),
			error_rate=round(error_rate, 2),
			uptime_seconds=round(uptime, 1)
		)
	
	def get_connection_details(self) -> List[Dict[str, Any]]:
		"""Get detailed connection information"""
		return [
			{
				"connection_id": conn.connection_id,
				"tenant_id": conn.tenant_id,
				"user_id": conn.user_id,
				"session_id": conn.session_id,
				"state": conn.state.value,
				"connected_at": conn.connected_at.isoformat(),
				"last_activity": conn.last_activity.isoformat(),
				"message_count": conn.message_count,
				"error_count": conn.error_count,
				"metadata": conn.metadata
			}
			for conn in self.active_connections.values()
		]
	
	async def cleanup(self) -> None:
		"""Cleanup streaming manager resources"""
		# Close all connections
		connection_ids = list(self.active_connections.keys())
		for connection_id in connection_ids:
			await self._cleanup_connection(connection_id)
		
		# Cleanup pipeline
		await self.pipeline.cleanup()
		
		logger.info(f"WebSocket streaming manager cleanup completed for tenant: {self.tenant_id}")

class WebSocketServer:
	"""WebSocket server for NLP streaming"""
	
	def __init__(self, tenant_id: str, host: str = "localhost", port: int = 8765, config: Dict[str, Any] = None):
		if not WEBSOCKETS_AVAILABLE:
			raise RuntimeError("WebSockets library not available - cannot create WebSocket server")
		
		self.tenant_id = tenant_id
		self.host = host
		self.port = port
		self.config = config or {}
		
		self.manager = WebSocketStreamingManager(tenant_id, config)
		self.server = None
	
	async def start_server(self) -> None:
		"""Start WebSocket server"""
		logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
		
		self.server = await websockets.serve(
			self.manager.handle_connection,
			self.host,
			self.port,
			max_size=self.config.get("max_message_size", 64 * 1024),
			max_queue=self.config.get("max_queue", 100)
		)
		
		logger.info(f"WebSocket server started successfully")
	
	async def stop_server(self) -> None:
		"""Stop WebSocket server"""
		if self.server:
			self.server.close()
			await self.server.wait_closed()
			logger.info("WebSocket server stopped")
		
		await self.manager.cleanup()
	
	def get_server_info(self) -> Dict[str, Any]:
		"""Get server information"""
		return {
			"host": self.host,
			"port": self.port,
			"tenant_id": self.tenant_id,
			"running": self.server is not None,
			"stats": self.manager.get_streaming_stats().__dict__,
			"websockets_available": WEBSOCKETS_AVAILABLE
		}

# Export main classes
__all__ = [
	"WebSocketStreamingManager", "WebSocketServer", "WebSocketMessage", 
	"WebSocketConnection", "StreamingStats", "MessageType", "ConnectionState"
]
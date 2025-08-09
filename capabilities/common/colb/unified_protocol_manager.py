"""
Unified Protocol Manager for APG Real-Time Collaboration

Orchestrates all communication protocols (WebRTC, WebSockets, MQTT, gRPC, 
Socket.IO, XMPP, SIP, RTMP) providing a single interface for multi-protocol 
real-time collaboration.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

# Import all protocol managers
try:
	from .mqtt_protocol import mqtt_protocol_manager, initialize_mqtt_protocol
	from .grpc_protocol import grpc_protocol_manager, initialize_grpc_protocol
	from .socketio_protocol import socketio_protocol_manager, initialize_socketio_protocol
	from .xmpp_protocol import xmpp_protocol_manager, initialize_xmpp_protocol
	from .sip_protocol import sip_protocol_manager, initialize_sip_protocol
	from .rtmp_protocol import rtmp_protocol_manager, initialize_rtmp_protocol
	from .websocket_manager import WebSocketManager
	from .webrtc_signaling import webrtc_signaling
except ImportError as e:
	print(f"Protocol import error: {e}")
	# Set to None if not available
	mqtt_protocol_manager = None
	grpc_protocol_manager = None
	socketio_protocol_manager = None
	xmpp_protocol_manager = None
	sip_protocol_manager = None
	rtmp_protocol_manager = None
	WebSocketManager = None
	webrtc_signaling = None

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
	"""Available communication protocols"""
	WEBSOCKET = "websocket"
	WEBRTC = "webrtc"
	MQTT = "mqtt"
	GRPC = "grpc"
	SOCKETIO = "socketio"
	XMPP = "xmpp"
	SIP = "sip"
	RTMP = "rtmp"


class MessagePriority(Enum):
	"""Message priority levels"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"


class CollaborationEventType(Enum):
	"""Collaboration event types"""
	USER_JOIN = "user_join"
	USER_LEAVE = "user_leave"
	CURSOR_MOVE = "cursor_move"
	TEXT_EDIT = "text_edit"
	FORM_UPDATE = "form_update"
	FILE_SHARE = "file_share"
	VOICE_CALL = "voice_call"
	VIDEO_CALL = "video_call"
	SCREEN_SHARE = "screen_share"
	PRESENCE_UPDATE = "presence_update"
	NOTIFICATION = "notification"
	WORKFLOW_EVENT = "workflow_event"
	IOT_SENSOR_DATA = "iot_sensor_data"


@dataclass
class UnifiedMessage:
	"""Unified message structure for all protocols"""
	message_id: str
	event_type: CollaborationEventType
	protocol: ProtocolType
	sender_id: str
	target_id: Optional[str] = None
	room_id: Optional[str] = None
	payload: Dict[str, Any] = field(default_factory=dict)
	priority: MessagePriority = MessagePriority.NORMAL
	timestamp: datetime = field(default_factory=datetime.utcnow)
	metadata: Dict[str, Any] = field(default_factory=dict)
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"message_id": self.message_id,
			"event_type": self.event_type.value,
			"protocol": self.protocol.value,
			"sender_id": self.sender_id,
			"target_id": self.target_id,
			"room_id": self.room_id,
			"payload": self.payload,
			"priority": self.priority.value,
			"timestamp": self.timestamp.isoformat(),
			"metadata": self.metadata
		}


@dataclass
class ProtocolConfiguration:
	"""Configuration for individual protocols"""
	protocol_type: ProtocolType
	enabled: bool = True
	config: Dict[str, Any] = field(default_factory=dict)
	priority: int = 1  # 1 = highest priority
	fallback_protocols: List[ProtocolType] = field(default_factory=list)
	max_retries: int = 3
	timeout_seconds: int = 30
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"protocol_type": self.protocol_type.value,
			"enabled": self.enabled,
			"config": self.config,
			"priority": self.priority,
			"fallback_protocols": [p.value for p in self.fallback_protocols],
			"max_retries": self.max_retries,
			"timeout_seconds": self.timeout_seconds
		}


@dataclass
class CollaborationSession:
	"""Collaboration session tracking"""
	session_id: str
	room_id: str
	participants: Set[str] = field(default_factory=set)
	active_protocols: Set[ProtocolType] = field(default_factory=set)
	session_type: str = "collaboration"  # collaboration, meeting, streaming
	created_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	metadata: Dict[str, Any] = field(default_factory=dict)
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"session_id": self.session_id,
			"room_id": self.room_id,
			"participants": list(self.participants),
			"active_protocols": [p.value for p in self.active_protocols],
			"session_type": self.session_type,
			"participant_count": len(self.participants),
			"created_at": self.created_at.isoformat(),
			"last_activity": self.last_activity.isoformat(),
			"metadata": self.metadata
		}


class UnifiedProtocolManager:
	"""Unified manager for all communication protocols"""
	
	def __init__(self):
		# Protocol managers
		self.protocol_managers: Dict[ProtocolType, Any] = {}
		self.protocol_configs: Dict[ProtocolType, ProtocolConfiguration] = {}
		self.protocol_status: Dict[ProtocolType, str] = {}
		
		# Session management
		self.active_sessions: Dict[str, CollaborationSession] = {}
		self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
		
		# Message handling
		self.message_handlers: Dict[CollaborationEventType, List[Callable]] = {}
		self.protocol_message_handlers: Dict[ProtocolType, List[Callable]] = {}
		self.message_queue: List[UnifiedMessage] = []
		self.max_queue_size = 10000
		
		# Routing and fallback
		self.protocol_routing: Dict[CollaborationEventType, List[ProtocolType]] = {}
		self.fallback_chains: Dict[ProtocolType, List[ProtocolType]] = {}
		
		# Statistics
		self.stats = {
			"messages_processed": 0,
			"messages_routed": 0,
			"messages_failed": 0,
			"protocol_failures": 0,
			"fallback_activations": 0,
			"active_sessions": 0,
			"total_participants": 0,
			"uptime_start": None
		}
		
		# Configuration
		self.auto_fallback = True
		self.message_persistence = True
		self.cross_protocol_routing = True
	
	async def initialize(self, protocol_configs: Dict[ProtocolType, Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Initialize all configured protocols"""
		try:
			self.stats["uptime_start"] = datetime.utcnow()
			
			# Set default configurations
			await self._setup_default_configurations()
			
			# Override with provided configurations
			if protocol_configs:
				for protocol_type, config in protocol_configs.items():
					if protocol_type in self.protocol_configs:
						self.protocol_configs[protocol_type].config.update(config)
			
			# Initialize protocols
			initialization_results = {}
			
			for protocol_type, config in self.protocol_configs.items():
				if config.enabled:
					result = await self._initialize_protocol(protocol_type, config)
					initialization_results[protocol_type.value] = result
					
					if result.get("status") in ["started", "connected"]:
						self.protocol_status[protocol_type] = "active"
					else:
						self.protocol_status[protocol_type] = "failed"
						logger.error(f"Failed to initialize {protocol_type.value}: {result}")
			
			# Setup protocol routing
			await self._setup_protocol_routing()
			
			# Setup fallback chains
			await self._setup_fallback_chains()
			
			# Start message processing
			asyncio.create_task(self._process_message_queue())
			
			active_protocols = [p.value for p, status in self.protocol_status.items() if status == "active"]
			
			logger.info(f"Unified Protocol Manager initialized with {len(active_protocols)} active protocols")
			
			return {
				"status": "initialized",
				"active_protocols": active_protocols,
				"protocol_results": initialization_results,
				"total_protocols": len(self.protocol_configs)
			}
			
		except Exception as e:
			logger.error(f"Failed to initialize Unified Protocol Manager: {e}")
			return {"error": f"Initialization failed: {str(e)}"}
	
	async def shutdown(self) -> Dict[str, Any]:
		"""Shutdown all protocols"""
		try:
			shutdown_results = {}
			
			# Terminate all sessions
			for session_id in list(self.active_sessions.keys()):
				await self.terminate_session(session_id)
			
			# Shutdown all protocols
			for protocol_type, manager in self.protocol_managers.items():
				try:
					if hasattr(manager, 'shutdown'):
						result = await manager.shutdown()
						shutdown_results[protocol_type.value] = result
					elif hasattr(manager, 'disconnect'):
						result = await manager.disconnect()
						shutdown_results[protocol_type.value] = result
					
					self.protocol_status[protocol_type] = "stopped"
					
				except Exception as e:
					logger.error(f"Error shutting down {protocol_type.value}: {e}")
					shutdown_results[protocol_type.value] = {"error": str(e)}
			
			logger.info("Unified Protocol Manager shut down")
			
			return {
				"status": "shutdown",
				"protocol_results": shutdown_results
			}
			
		except Exception as e:
			logger.error(f"Error during shutdown: {e}")
			return {"error": f"Shutdown failed: {str(e)}"}
	
	async def _setup_default_configurations(self):
		"""Setup default protocol configurations"""
		# WebSocket configuration
		self.protocol_configs[ProtocolType.WEBSOCKET] = ProtocolConfiguration(
			protocol_type=ProtocolType.WEBSOCKET,
			enabled=True,
			config={"host": "localhost", "port": 8765},
			priority=1,
			fallback_protocols=[ProtocolType.SOCKETIO, ProtocolType.GRPC]
		)
		
		# WebRTC configuration  
		self.protocol_configs[ProtocolType.WEBRTC] = ProtocolConfiguration(
			protocol_type=ProtocolType.WEBRTC,
			enabled=True,
			config={},
			priority=1,
			fallback_protocols=[ProtocolType.WEBSOCKET, ProtocolType.SIP]
		)
		
		# MQTT configuration
		self.protocol_configs[ProtocolType.MQTT] = ProtocolConfiguration(
			protocol_type=ProtocolType.MQTT,
			enabled=True,
			config={"broker_host": "localhost", "broker_port": 1883},
			priority=2,
			fallback_protocols=[ProtocolType.GRPC, ProtocolType.WEBSOCKET]
		)
		
		# gRPC configuration
		self.protocol_configs[ProtocolType.GRPC] = ProtocolConfiguration(
			protocol_type=ProtocolType.GRPC,
			enabled=True,
			config={"host": "localhost", "port": 50051},
			priority=2,
			fallback_protocols=[ProtocolType.WEBSOCKET, ProtocolType.MQTT]
		)
		
		# Socket.IO configuration
		self.protocol_configs[ProtocolType.SOCKETIO] = ProtocolConfiguration(
			protocol_type=ProtocolType.SOCKETIO,
			enabled=True,
			config={"host": "localhost", "port": 3000},
			priority=3,
			fallback_protocols=[ProtocolType.WEBSOCKET, ProtocolType.GRPC]
		)
		
		# XMPP configuration
		self.protocol_configs[ProtocolType.XMPP] = ProtocolConfiguration(
			protocol_type=ProtocolType.XMPP,
			enabled=False,  # Requires credentials
			config={"jid": "user@example.com", "password": "password"},
			priority=4,
			fallback_protocols=[ProtocolType.MQTT, ProtocolType.WEBSOCKET]
		)
		
		# SIP configuration
		self.protocol_configs[ProtocolType.SIP] = ProtocolConfiguration(
			protocol_type=ProtocolType.SIP,
			enabled=True,
			config={"local_host": "0.0.0.0", "local_port": 5060},
			priority=3,
			fallback_protocols=[ProtocolType.WEBRTC, ProtocolType.WEBSOCKET]
		)
		
		# RTMP configuration
		self.protocol_configs[ProtocolType.RTMP] = ProtocolConfiguration(
			protocol_type=ProtocolType.RTMP,
			enabled=True,
			config={"host": "0.0.0.0", "port": 1935},
			priority=4,
			fallback_protocols=[ProtocolType.WEBRTC, ProtocolType.WEBSOCKET]
		)
	
	async def _initialize_protocol(self, protocol_type: ProtocolType, config: ProtocolConfiguration) -> Dict[str, Any]:
		"""Initialize individual protocol"""
		try:
			if protocol_type == ProtocolType.WEBSOCKET:
				if WebSocketManager:
					manager = WebSocketManager(**config.config)
					result = await manager.initialize()
					self.protocol_managers[protocol_type] = manager
					return result
				
			elif protocol_type == ProtocolType.WEBRTC:
				if webrtc_signaling:
					# WebRTC is initialized elsewhere
					self.protocol_managers[protocol_type] = webrtc_signaling
					return {"status": "active", "note": "WebRTC already initialized"}
				
			elif protocol_type == ProtocolType.MQTT:
				if initialize_mqtt_protocol:
					result = await initialize_mqtt_protocol(**config.config)
					self.protocol_managers[protocol_type] = mqtt_protocol_manager
					return result
					
			elif protocol_type == ProtocolType.GRPC:
				if initialize_grpc_protocol:
					result = await initialize_grpc_protocol(**config.config)
					self.protocol_managers[protocol_type] = grpc_protocol_manager
					return result
					
			elif protocol_type == ProtocolType.SOCKETIO:
				if initialize_socketio_protocol:
					result = await initialize_socketio_protocol(**config.config)
					self.protocol_managers[protocol_type] = socketio_protocol_manager
					return result
					
			elif protocol_type == ProtocolType.XMPP:
				if initialize_xmpp_protocol and config.config.get("jid") and config.config.get("password"):
					result = await initialize_xmpp_protocol(
						config.config["jid"],
						config.config["password"],
						config.config.get("server")
					)
					self.protocol_managers[protocol_type] = xmpp_protocol_manager
					return result
					
			elif protocol_type == ProtocolType.SIP:
				if initialize_sip_protocol:
					result = await initialize_sip_protocol(**config.config)
					self.protocol_managers[protocol_type] = sip_protocol_manager
					return result
					
			elif protocol_type == ProtocolType.RTMP:
				if initialize_rtmp_protocol:
					result = await initialize_rtmp_protocol(**config.config)
					self.protocol_managers[protocol_type] = rtmp_protocol_manager
					return result
			
			return {"error": f"Protocol {protocol_type.value} not available or not configured"}
			
		except Exception as e:
			logger.error(f"Error initializing {protocol_type.value}: {e}")
			return {"error": f"Initialization failed: {str(e)}"}
	
	async def _setup_protocol_routing(self):
		"""Setup routing rules for different event types"""
		self.protocol_routing = {
			CollaborationEventType.USER_JOIN: [ProtocolType.WEBSOCKET, ProtocolType.SOCKETIO, ProtocolType.MQTT],
			CollaborationEventType.USER_LEAVE: [ProtocolType.WEBSOCKET, ProtocolType.SOCKETIO, ProtocolType.MQTT],
			CollaborationEventType.CURSOR_MOVE: [ProtocolType.WEBSOCKET, ProtocolType.SOCKETIO],
			CollaborationEventType.TEXT_EDIT: [ProtocolType.WEBSOCKET, ProtocolType.SOCKETIO, ProtocolType.GRPC],
			CollaborationEventType.FORM_UPDATE: [ProtocolType.WEBSOCKET, ProtocolType.GRPC, ProtocolType.MQTT],
			CollaborationEventType.FILE_SHARE: [ProtocolType.WEBRTC, ProtocolType.GRPC, ProtocolType.WEBSOCKET],
			CollaborationEventType.VOICE_CALL: [ProtocolType.WEBRTC, ProtocolType.SIP],
			CollaborationEventType.VIDEO_CALL: [ProtocolType.WEBRTC, ProtocolType.SIP, ProtocolType.RTMP],
			CollaborationEventType.SCREEN_SHARE: [ProtocolType.WEBRTC, ProtocolType.RTMP],
			CollaborationEventType.PRESENCE_UPDATE: [ProtocolType.WEBSOCKET, ProtocolType.XMPP, ProtocolType.MQTT],
			CollaborationEventType.NOTIFICATION: [ProtocolType.WEBSOCKET, ProtocolType.SOCKETIO, ProtocolType.MQTT],
			CollaborationEventType.WORKFLOW_EVENT: [ProtocolType.GRPC, ProtocolType.MQTT, ProtocolType.WEBSOCKET],
			CollaborationEventType.IOT_SENSOR_DATA: [ProtocolType.MQTT, ProtocolType.GRPC]
		}
	
	async def _setup_fallback_chains(self):
		"""Setup fallback chains for protocols"""
		for protocol_type, config in self.protocol_configs.items():
			self.fallback_chains[protocol_type] = config.fallback_protocols.copy()
	
	async def create_session(self, room_id: str, session_type: str = "collaboration", 
							 creator_id: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Create new collaboration session"""
		try:
			session_id = str(uuid.uuid4())
			
			session = CollaborationSession(
				session_id=session_id,
				room_id=room_id,
				session_type=session_type,
				metadata=metadata or {}
			)
			
			if creator_id:
				session.participants.add(creator_id)
				
				# Track user sessions
				if creator_id not in self.user_sessions:
					self.user_sessions[creator_id] = set()
				self.user_sessions[creator_id].add(session_id)
			
			self.active_sessions[session_id] = session
			
			# Update statistics
			self.stats["active_sessions"] = len(self.active_sessions)
			self.stats["total_participants"] = sum(len(s.participants) for s in self.active_sessions.values())
			
			logger.info(f"Created collaboration session: {session_id} in room {room_id}")
			
			return {
				"status": "created",
				"session_id": session_id,
				"room_id": room_id,
				"session_type": session_type
			}
			
		except Exception as e:
			logger.error(f"Failed to create session: {e}")
			return {"error": f"Session creation failed: {str(e)}"}
	
	async def join_session(self, session_id: str, user_id: str, 
						   preferred_protocols: List[ProtocolType] = None) -> Dict[str, Any]:
		"""Join collaboration session"""
		try:
			if session_id not in self.active_sessions:
				return {"error": "Session not found"}
			
			session = self.active_sessions[session_id]
			session.participants.add(user_id)
			session.last_activity = datetime.utcnow()
			
			# Track user sessions
			if user_id not in self.user_sessions:
				self.user_sessions[user_id] = set()
			self.user_sessions[user_id].add(session_id)
			
			# Determine active protocols for user
			active_protocols = preferred_protocols or [ProtocolType.WEBSOCKET, ProtocolType.WEBRTC]
			session.active_protocols.update(active_protocols)
			
			# Send user join event
			join_message = UnifiedMessage(
				message_id=str(uuid.uuid4()),
				event_type=CollaborationEventType.USER_JOIN,
				protocol=ProtocolType.WEBSOCKET,  # Primary protocol
				sender_id=user_id,
				room_id=session.room_id,
				payload={
					"session_id": session_id,
					"user_id": user_id,
					"active_protocols": [p.value for p in active_protocols]
				}
			)
			
			await self.route_message(join_message)
			
			# Update statistics
			self.stats["total_participants"] = sum(len(s.participants) for s in self.active_sessions.values())
			
			logger.info(f"User {user_id} joined session {session_id}")
			
			return {
				"status": "joined",
				"session_id": session_id,
				"participant_count": len(session.participants),
				"active_protocols": [p.value for p in active_protocols]
			}
			
		except Exception as e:
			logger.error(f"Failed to join session: {e}")
			return {"error": f"Join session failed: {str(e)}"}
	
	async def leave_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
		"""Leave collaboration session"""
		try:
			if session_id not in self.active_sessions:
				return {"error": "Session not found"}
			
			session = self.active_sessions[session_id]
			session.participants.discard(user_id)
			session.last_activity = datetime.utcnow()
			
			# Remove from user sessions
			if user_id in self.user_sessions:
				self.user_sessions[user_id].discard(session_id)
				if not self.user_sessions[user_id]:
					del self.user_sessions[user_id]
			
			# Send user leave event
			leave_message = UnifiedMessage(
				message_id=str(uuid.uuid4()),
				event_type=CollaborationEventType.USER_LEAVE,
				protocol=ProtocolType.WEBSOCKET,
				sender_id=user_id,
				room_id=session.room_id,
				payload={
					"session_id": session_id,
					"user_id": user_id
				}
			)
			
			await self.route_message(leave_message)
			
			# Remove session if empty
			if not session.participants:
				await self.terminate_session(session_id)
			
			# Update statistics
			self.stats["total_participants"] = sum(len(s.participants) for s in self.active_sessions.values())
			
			logger.info(f"User {user_id} left session {session_id}")
			
			return {
				"status": "left",
				"session_id": session_id,
				"participant_count": len(session.participants)
			}
			
		except Exception as e:
			logger.error(f"Failed to leave session: {e}")
			return {"error": f"Leave session failed: {str(e)}"}
	
	async def terminate_session(self, session_id: str) -> Dict[str, Any]:
		"""Terminate collaboration session"""
		try:
			if session_id not in self.active_sessions:
				return {"error": "Session not found"}
			
			session = self.active_sessions[session_id]
			
			# Remove users from session tracking
			for user_id in session.participants:
				if user_id in self.user_sessions:
					self.user_sessions[user_id].discard(session_id)
					if not self.user_sessions[user_id]:
						del self.user_sessions[user_id]
			
			# Remove session
			del self.active_sessions[session_id]
			
			# Update statistics
			self.stats["active_sessions"] = len(self.active_sessions)
			self.stats["total_participants"] = sum(len(s.participants) for s in self.active_sessions.values())
			
			logger.info(f"Terminated session {session_id}")
			
			return {
				"status": "terminated",
				"session_id": session_id
			}
			
		except Exception as e:
			logger.error(f"Failed to terminate session: {e}")
			return {"error": f"Session termination failed: {str(e)}"}
	
	async def send_message(self, message: UnifiedMessage) -> Dict[str, Any]:
		"""Send message through appropriate protocol"""
		try:
			# Add to message queue for processing
			if len(self.message_queue) < self.max_queue_size:
				self.message_queue.append(message)
				return {"status": "queued", "message_id": message.message_id}
			else:
				return {"error": "Message queue full"}
			
		except Exception as e:
			logger.error(f"Failed to send message: {e}")
			return {"error": f"Send message failed: {str(e)}"}
	
	async def route_message(self, message: UnifiedMessage) -> Dict[str, Any]:
		"""Route message to appropriate protocols"""
		try:
			# Get routing protocols for event type
			target_protocols = self.protocol_routing.get(message.event_type, [ProtocolType.WEBSOCKET])
			
			# Filter by active protocols
			active_protocols = [p for p in target_protocols if self.protocol_status.get(p) == "active"]
			
			if not active_protocols:
				# Use fallback protocols
				active_protocols = [ProtocolType.WEBSOCKET]  # Default fallback
			
			routing_results = {}
			success_count = 0
			
			for protocol in active_protocols:
				try:
					result = await self._send_via_protocol(message, protocol)
					routing_results[protocol.value] = result
					
					if result.get("status") in ["sent", "published", "broadcast"]:
						success_count += 1
						
				except Exception as e:
					logger.error(f"Error routing message via {protocol.value}: {e}")
					routing_results[protocol.value] = {"error": str(e)}
					
					# Try fallback protocols
					if self.auto_fallback and protocol in self.fallback_chains:
						await self._try_fallback_protocols(message, protocol)
			
			# Update statistics
			self.stats["messages_routed"] += 1
			if success_count == 0:
				self.stats["messages_failed"] += 1
			
			return {
				"status": "routed" if success_count > 0 else "failed",
				"message_id": message.message_id,
				"protocols_used": list(routing_results.keys()),
				"success_count": success_count,
				"results": routing_results
			}
			
		except Exception as e:
			logger.error(f"Failed to route message: {e}")
			self.stats["messages_failed"] += 1
			return {"error": f"Message routing failed: {str(e)}"}
	
	async def _send_via_protocol(self, message: UnifiedMessage, protocol: ProtocolType) -> Dict[str, Any]:
		"""Send message via specific protocol"""
		try:
			manager = self.protocol_managers.get(protocol)
			if not manager:
				return {"error": f"Protocol manager not available: {protocol.value}"}
			
			# Convert unified message to protocol-specific format and send
			if protocol == ProtocolType.WEBSOCKET:
				if hasattr(manager, 'broadcast_message'):
					return await manager.broadcast_message(message.payload)
					
			elif protocol == ProtocolType.MQTT:
				if hasattr(manager, 'publish_collaboration_event'):
					return await manager.publish_collaboration_event(
						message.event_type.value, 
						message.payload
					)
					
			elif protocol == ProtocolType.GRPC:
				if hasattr(manager, 'send_collaboration_event'):
					return await manager.send_collaboration_event(
						"default_client",
						message.event_type.value,
						message.payload
					)
					
			elif protocol == ProtocolType.SOCKETIO:
				if hasattr(manager, 'broadcast'):
					return await manager.broadcast(
						message.event_type.value,
						message.payload
					)
					
			elif protocol == ProtocolType.XMPP:
				if hasattr(manager, 'send_message') and message.target_id:
					return await manager.send_message(
						message.target_id,
						json.dumps(message.payload)
					)
					
			elif protocol == ProtocolType.SIP:
				# SIP is primarily for voice/video calls
				if message.event_type in [CollaborationEventType.VOICE_CALL, CollaborationEventType.VIDEO_CALL]:
					if hasattr(manager, 'initiate_call') and message.target_id:
						return await manager.initiate_call(
							message.sender_id,
							message.target_id
						)
			
			elif protocol == ProtocolType.RTMP:
				# RTMP is primarily for streaming
				if message.event_type == CollaborationEventType.SCREEN_SHARE:
					if hasattr(manager, 'create_stream'):
						return await manager.create_stream(
							message.payload.get("stream_key", "default_stream")
						)
			
			return {"status": "sent", "protocol": protocol.value}
			
		except Exception as e:
			logger.error(f"Error sending via {protocol.value}: {e}")
			return {"error": f"Send failed: {str(e)}"}
	
	async def _try_fallback_protocols(self, message: UnifiedMessage, failed_protocol: ProtocolType):
		"""Try fallback protocols when primary fails"""
		try:
			fallback_protocols = self.fallback_chains.get(failed_protocol, [])
			
			for fallback_protocol in fallback_protocols:
				if self.protocol_status.get(fallback_protocol) == "active":
					try:
						result = await self._send_via_protocol(message, fallback_protocol)
						
						if result.get("status") in ["sent", "published", "broadcast"]:
							self.stats["fallback_activations"] += 1
							logger.info(f"Successfully used fallback protocol {fallback_protocol.value}")
							break
							
					except Exception as e:
						logger.error(f"Fallback protocol {fallback_protocol.value} also failed: {e}")
						continue
			
		except Exception as e:
			logger.error(f"Error in fallback protocol handling: {e}")
	
	async def _process_message_queue(self):
		"""Process queued messages"""
		while True:
			try:
				if self.message_queue:
					# Process messages in batches
					batch_size = min(10, len(self.message_queue))
					messages_to_process = self.message_queue[:batch_size]
					self.message_queue = self.message_queue[batch_size:]
					
					for message in messages_to_process:
						await self.route_message(message)
						self.stats["messages_processed"] += 1
				
				await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
				
			except Exception as e:
				logger.error(f"Error processing message queue: {e}")
				await asyncio.sleep(1)  # Longer delay on error
	
	def register_message_handler(self, event_type: CollaborationEventType, handler: Callable):
		"""Register message handler for specific event type"""
		if event_type not in self.message_handlers:
			self.message_handlers[event_type] = []
		self.message_handlers[event_type].append(handler)
		
		logger.info(f"Registered message handler for event type: {event_type.value}")
	
	def register_protocol_handler(self, protocol: ProtocolType, handler: Callable):
		"""Register handler for specific protocol"""
		if protocol not in self.protocol_message_handlers:
			self.protocol_message_handlers[protocol] = []
		self.protocol_message_handlers[protocol].append(handler)
		
		logger.info(f"Registered protocol handler for: {protocol.value}")
	
	def get_active_sessions(self) -> List[Dict[str, Any]]:
		"""Get list of active collaboration sessions"""
		return [session.to_dict() for session in self.active_sessions.values()]
	
	def get_protocol_status(self) -> Dict[str, Any]:
		"""Get status of all protocols"""
		return {
			protocol.value: {
				"status": self.protocol_status.get(protocol, "unknown"),
				"enabled": self.protocol_configs.get(protocol, {}).enabled if protocol in self.protocol_configs else False,
				"priority": self.protocol_configs.get(protocol, {}).priority if protocol in self.protocol_configs else 999
			}
			for protocol in ProtocolType
		}
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get unified protocol manager statistics"""
		uptime_seconds = 0
		if self.stats["uptime_start"]:
			uptime_seconds = int((datetime.utcnow() - self.stats["uptime_start"]).total_seconds())
		
		protocol_stats = {}
		for protocol, manager in self.protocol_managers.items():
			if hasattr(manager, 'get_statistics'):
				try:
					protocol_stats[protocol.value] = manager.get_statistics()
				except Exception as e:
					protocol_stats[protocol.value] = {"error": str(e)}
		
		return {
			**self.stats,
			"uptime_seconds": uptime_seconds,
			"queue_size": len(self.message_queue),
			"active_protocols": len([p for p in self.protocol_status.values() if p == "active"]),
			"protocol_status": self.get_protocol_status(),
			"protocol_statistics": protocol_stats,
			"uptime_start": self.stats["uptime_start"].isoformat() if self.stats["uptime_start"] else None
		}


# Global unified protocol manager instance
unified_protocol_manager = None


async def initialize_unified_protocols(protocol_configs: Dict[ProtocolType, Dict[str, Any]] = None) -> Dict[str, Any]:
	"""Initialize global unified protocol manager"""
	global unified_protocol_manager
	
	unified_protocol_manager = UnifiedProtocolManager()
	
	result = await unified_protocol_manager.initialize(protocol_configs)
	
	return result


def get_unified_manager() -> Optional[UnifiedProtocolManager]:
	"""Get global unified protocol manager"""
	return unified_protocol_manager


if __name__ == "__main__":
	# Test unified protocol manager
	async def test_unified_protocols():
		print("Testing Unified Protocol Manager...")
		
		# Initialize with custom configurations
		protocol_configs = {
			ProtocolType.WEBSOCKET: {"host": "localhost", "port": 8765},
			ProtocolType.MQTT: {"broker_host": "localhost", "broker_port": 1883},
			ProtocolType.GRPC: {"host": "localhost", "port": 50051},
			ProtocolType.SOCKETIO: {"host": "localhost", "port": 3000},
			ProtocolType.SIP: {"local_host": "0.0.0.0", "local_port": 5060},
			ProtocolType.RTMP: {"host": "0.0.0.0", "port": 1935}
		}
		
		# Initialize unified manager
		result = await initialize_unified_protocols(protocol_configs)
		print(f"Unified initialization result: {result}")
		
		if result.get("status") == "initialized":
			manager = get_unified_manager()
			
			# Create test session
			session_result = await manager.create_session(
				"test_room", 
				"collaboration", 
				"user1"
			)
			print(f"Session creation result: {session_result}")
			
			if session_result.get("status") == "created":
				session_id = session_result["session_id"]
				
				# Join session with another user
				join_result = await manager.join_session(
					session_id, 
					"user2",
					[ProtocolType.WEBSOCKET, ProtocolType.WEBRTC]
				)
				print(f"Join session result: {join_result}")
				
				# Send test message
				test_message = UnifiedMessage(
					message_id=str(uuid.uuid4()),
					event_type=CollaborationEventType.TEXT_EDIT,
					protocol=ProtocolType.WEBSOCKET,
					sender_id="user1",
					room_id="test_room",
					payload={
						"field_name": "description",
						"operation": "insert",
						"content": "Hello, collaborative world!"
					}
				)
				
				send_result = await manager.send_message(test_message)
				print(f"Send message result: {send_result}")
				
				# Wait for message processing
				await asyncio.sleep(2)
			
			# Get statistics
			stats = manager.get_statistics()
			print(f"Unified statistics: {stats}")
			
			# Get protocol status
			protocol_status = manager.get_protocol_status()
			print(f"Protocol status: {protocol_status}")
			
			# Get active sessions
			sessions = manager.get_active_sessions()
			print(f"Active sessions: {len(sessions)}")
			
			# Wait a bit longer
			await asyncio.sleep(3)
			
			# Shutdown
			shutdown_result = await manager.shutdown()
			print(f"Shutdown result: {shutdown_result}")
		
		print("✅ Unified Protocol Manager test completed")
	
	asyncio.run(test_unified_protocols())
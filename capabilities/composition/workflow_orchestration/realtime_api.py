#!/usr/bin/env python3
"""
APG Workflow Orchestration Real-time API Features

WebSocket API, event streaming, real-time collaboration, and live updates
for the workflow orchestration system.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Union
from uuid import uuid4
from dataclasses import dataclass, asdict
from enum import Enum
import weakref

import aioredis
from aiohttp import web, WSMsgType
import socketio
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

# APG Framework imports
from apg.framework.auth_rbac import APGAuth, APGUser
from apg.framework.base_service import APGBaseService
from apg.framework.messaging import APGEventBus, APGMessage
from apg.framework.security import APGSecurity
from apg.framework.audit_compliance import APGAuditLogger, AuditEvent

# Local imports
from .models import WorkflowStatus, TaskStatus
from .database import WorkflowDB, WorkflowInstanceDB, TaskExecutionDB


logger = logging.getLogger(__name__)


class EventType(str, Enum):
	"""Types of real-time events."""
	
	# Workflow events
	WORKFLOW_CREATED = "workflow.created"
	WORKFLOW_UPDATED = "workflow.updated"
	WORKFLOW_DELETED = "workflow.deleted"
	WORKFLOW_EXECUTED = "workflow.executed"
	
	# Workflow instance events
	INSTANCE_CREATED = "instance.created"
	INSTANCE_STARTED = "instance.started"
	INSTANCE_COMPLETED = "instance.completed"
	INSTANCE_FAILED = "instance.failed"
	INSTANCE_CANCELLED = "instance.cancelled"
	INSTANCE_PROGRESS = "instance.progress"
	
	# Task execution events
	TASK_STARTED = "task.started"
	TASK_COMPLETED = "task.completed"
	TASK_FAILED = "task.failed"
	TASK_RETRYING = "task.retrying"
	TASK_PROGRESS = "task.progress"
	
	# Collaboration events
	USER_JOINED = "collaboration.user_joined"
	USER_LEFT = "collaboration.user_left"
	USER_TYPING = "collaboration.user_typing"
	CURSOR_MOVED = "collaboration.cursor_moved"
	SELECTION_CHANGED = "collaboration.selection_changed"
	
	# Designer events
	COMPONENT_ADDED = "designer.component_added"
	COMPONENT_UPDATED = "designer.component_updated"
	COMPONENT_DELETED = "designer.component_deleted"
	CONNECTION_CREATED = "designer.connection_created"
	CONNECTION_DELETED = "designer.connection_deleted"
	CANVAS_UPDATED = "designer.canvas_updated"
	
	# System events
	SYSTEM_STATUS = "system.status"
	INTEGRATION_STATUS = "integration.status"
	METRICS_UPDATE = "metrics.update"


class SubscriptionType(str, Enum):
	"""Types of event subscriptions."""
	
	WORKFLOW = "workflow"
	WORKFLOW_INSTANCE = "workflow_instance"
	TASK_EXECUTION = "task_execution"
	COLLABORATION_SESSION = "collaboration_session"
	TENANT = "tenant"
	USER = "user"
	SYSTEM = "system"


@dataclass
class RealTimeEvent:
	"""Real-time event data structure."""
	
	id: str
	event_type: EventType
	source: str
	resource_type: str
	resource_id: str
	tenant_id: str
	user_id: Optional[str] = None
	session_id: Optional[str] = None
	data: Dict[str, Any] = None
	timestamp: datetime = None
	ttl_seconds: int = 300  # 5 minutes default TTL
	
	def __post_init__(self):
		if self.timestamp is None:
			self.timestamp = datetime.utcnow()
		if self.data is None:
			self.data = {}


@dataclass
class EventSubscription:
	"""Event subscription configuration."""
	
	id: str
	user_id: str
	session_id: str
	tenant_id: str
	subscription_type: SubscriptionType
	resource_id: Optional[str] = None
	event_types: List[EventType] = None
	filters: Dict[str, Any] = None
	created_at: datetime = None
	last_active: datetime = None
	
	def __post_init__(self):
		if self.created_at is None:
			self.created_at = datetime.utcnow()
		if self.last_active is None:
			self.last_active = datetime.utcnow()
		if self.event_types is None:
			self.event_types = []
		if self.filters is None:
			self.filters = {}


@dataclass
class CollaborationSession:
	"""Real-time collaboration session."""
	
	id: str
	resource_type: str
	resource_id: str
	tenant_id: str
	participants: Set[str]
	created_at: datetime
	last_activity: datetime
	metadata: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.metadata is None:
			self.metadata = {}


@dataclass
class UserPresence:
	"""User presence information."""
	
	user_id: str
	session_id: str
	tenant_id: str
	status: str = "online"  # online, away, busy, offline
	last_seen: datetime = None
	current_resource: Optional[str] = None
	cursor_position: Optional[Dict[str, Any]] = None
	metadata: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.last_seen is None:
			self.last_seen = datetime.utcnow()
		if self.metadata is None:
			self.metadata = {}


class WebSocketConnection:
	"""WebSocket connection wrapper."""
	
	def __init__(self, ws: web.WebSocketResponse, user_id: str, session_id: str, tenant_id: str):
		self.ws = ws
		self.user_id = user_id
		self.session_id = session_id
		self.tenant_id = tenant_id
		self.subscriptions: Set[str] = set()
		self.last_ping = datetime.utcnow()
		self.metadata: Dict[str, Any] = {}
	
	async def send_event(self, event: RealTimeEvent):
		"""Send event to this connection."""
		try:
			message = {
				'type': 'event',
				'event': asdict(event)
			}
			await self.ws.send_str(json.dumps(message))
		except Exception as e:
			logger.error(f"Failed to send event to connection {self.session_id}: {str(e)}")
	
	async def send_message(self, message_type: str, data: Dict[str, Any]):
		"""Send custom message to this connection."""
		try:
			message = {
				'type': message_type,
				'data': data,
				'timestamp': datetime.utcnow().isoformat()
			}
			await self.ws.send_str(json.dumps(message))
		except Exception as e:
			logger.error(f"Failed to send message to connection {self.session_id}: {str(e)}")
	
	def is_alive(self) -> bool:
		"""Check if connection is still alive."""
		return not self.ws.closed


class EventStream:
	"""Event streaming manager."""
	
	def __init__(self, redis_client: aioredis.Redis):
		self.redis = redis_client
		self.subscribers: Dict[str, Set[Callable]] = {}
		self.event_history: Dict[str, List[RealTimeEvent]] = {}
		self.max_history_size = 1000
	
	async def publish(self, event: RealTimeEvent):
		"""Publish event to stream."""
		try:
			# Store in Redis stream
			stream_key = f"events:{event.tenant_id}:{event.resource_type}"
			event_data = asdict(event)
			event_data['timestamp'] = event.timestamp.isoformat()
			
			await self.redis.xadd(
				stream_key,
				event_data,
				maxlen=self.max_history_size
			)
			
			# Publish to Redis pub/sub for real-time delivery
			channel = f"realtime:{event.tenant_id}"
			await self.redis.publish(channel, json.dumps(event_data))
			
			# Store in memory for quick access
			history_key = f"{event.tenant_id}:{event.resource_type}:{event.resource_id}"
			if history_key not in self.event_history:
				self.event_history[history_key] = []
			
			self.event_history[history_key].append(event)
			
			# Limit history size
			if len(self.event_history[history_key]) > 100:
				self.event_history[history_key] = self.event_history[history_key][-100:]
			
			# Notify local subscribers
			for subscriber in self.subscribers.get(event.event_type, set()):
				try:
					if asyncio.iscoroutinefunction(subscriber):
						await subscriber(event)
					else:
						subscriber(event)
				except Exception as e:
					logger.error(f"Error in event subscriber: {str(e)}")
			
		except Exception as e:
			logger.error(f"Failed to publish event: {str(e)}")
	
	async def subscribe(self, event_type: EventType, callback: Callable):
		"""Subscribe to events of a specific type."""
		if event_type not in self.subscribers:
			self.subscribers[event_type] = set()
		self.subscribers[event_type].add(callback)
	
	async def unsubscribe(self, event_type: EventType, callback: Callable):
		"""Unsubscribe from events."""
		if event_type in self.subscribers:
			self.subscribers[event_type].discard(callback)
	
	async def get_event_history(self, tenant_id: str, resource_type: str, 
								resource_id: str, limit: int = 50) -> List[RealTimeEvent]:
		"""Get recent event history."""
		try:
			# Try memory first
			history_key = f"{tenant_id}:{resource_type}:{resource_id}"
			if history_key in self.event_history:
				return self.event_history[history_key][-limit:]
			
			# Fallback to Redis
			stream_key = f"events:{tenant_id}:{resource_type}"
			events = await self.redis.xrevrange(stream_key, count=limit)
			
			result = []
			for event_id, fields in events:
				if fields.get('resource_id') == resource_id:
					event_data = dict(fields)
					event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
					result.append(RealTimeEvent(**event_data))
			
			return result
			
		except Exception as e:
			logger.error(f"Failed to get event history: {str(e)}")
			return []


class RealTimeCollaboration:
	"""Real-time collaboration manager."""
	
	def __init__(self, event_stream: EventStream):
		self.event_stream = event_stream
		self.sessions: Dict[str, CollaborationSession] = {}
		self.user_presence: Dict[str, UserPresence] = {}
		self.session_participants: Dict[str, Set[str]] = {}  # session_id -> set of user_ids
	
	async def create_session(self, resource_type: str, resource_id: str, 
							tenant_id: str, user_id: str) -> str:
		"""Create or join a collaboration session."""
		session_id = f"{resource_type}:{resource_id}:{tenant_id}"
		
		if session_id not in self.sessions:
			self.sessions[session_id] = CollaborationSession(
				id=session_id,
				resource_type=resource_type,
				resource_id=resource_id,
				tenant_id=tenant_id,
				participants=set(),
				created_at=datetime.utcnow(),
				last_activity=datetime.utcnow()
			)
			self.session_participants[session_id] = set()
		
		# Add user to session
		self.sessions[session_id].participants.add(user_id)
		self.sessions[session_id].last_activity = datetime.utcnow()
		self.session_participants[session_id].add(user_id)
		
		# Update user presence
		self.user_presence[user_id] = UserPresence(
			user_id=user_id,
			session_id=session_id,
			tenant_id=tenant_id,
			current_resource=f"{resource_type}:{resource_id}"
		)
		
		# Publish user joined event
		event = RealTimeEvent(
			id=uuid7str(),
			event_type=EventType.USER_JOINED,
			source="collaboration",
			resource_type=resource_type,
			resource_id=resource_id,
			tenant_id=tenant_id,
			user_id=user_id,
			session_id=session_id,
			data={
				'session_id': session_id,
				'participant_count': len(self.sessions[session_id].participants)
			}
		)
		await self.event_stream.publish(event)
		
		return session_id
	
	async def leave_session(self, session_id: str, user_id: str):
		"""Remove user from collaboration session."""
		if session_id in self.sessions:
			self.sessions[session_id].participants.discard(user_id)
			self.session_participants[session_id].discard(user_id)
			
			# Update user presence
			if user_id in self.user_presence:
				self.user_presence[user_id].status = "offline"
				self.user_presence[user_id].current_resource = None
			
			# Publish user left event
			event = RealTimeEvent(
				id=uuid7str(),
				event_type=EventType.USER_LEFT,
				source="collaboration",
				resource_type=self.sessions[session_id].resource_type,
				resource_id=self.sessions[session_id].resource_id,
				tenant_id=self.sessions[session_id].tenant_id,
				user_id=user_id,
				session_id=session_id,
				data={
					'session_id': session_id,
					'participant_count': len(self.sessions[session_id].participants)
				}
			)
			await self.event_stream.publish(event)
			
			# Clean up empty session
			if not self.sessions[session_id].participants:
				del self.sessions[session_id]
				del self.session_participants[session_id]
	
	async def update_cursor(self, session_id: str, user_id: str, cursor_data: Dict[str, Any]):
		"""Update user cursor position."""
		if user_id in self.user_presence:
			self.user_presence[user_id].cursor_position = cursor_data
			self.user_presence[user_id].last_seen = datetime.utcnow()
		
		if session_id in self.sessions:
			session = self.sessions[session_id]
			event = RealTimeEvent(
				id=uuid7str(),
				event_type=EventType.CURSOR_MOVED,
				source="collaboration",
				resource_type=session.resource_type,
				resource_id=session.resource_id,
				tenant_id=session.tenant_id,
				user_id=user_id,
				session_id=session_id,
				data=cursor_data
			)
			await self.event_stream.publish(event)
	
	async def broadcast_to_session(self, session_id: str, event: RealTimeEvent, 
								   exclude_user: Optional[str] = None):
		"""Broadcast event to all participants in a session."""
		if session_id in self.session_participants:
			for user_id in self.session_participants[session_id]:
				if exclude_user and user_id == exclude_user:
					continue
				
				# Create targeted event
				targeted_event = RealTimeEvent(
					id=event.id,
					event_type=event.event_type,
					source=event.source,
					resource_type=event.resource_type,
					resource_id=event.resource_id,
					tenant_id=event.tenant_id,
					user_id=user_id,
					session_id=session_id,
					data=event.data,
					timestamp=event.timestamp
				)
				await self.event_stream.publish(targeted_event)
	
	def get_session_participants(self, session_id: str) -> List[UserPresence]:
		"""Get all participants in a session."""
		if session_id not in self.session_participants:
			return []
		
		participants = []
		for user_id in self.session_participants[session_id]:
			if user_id in self.user_presence:
				participants.append(self.user_presence[user_id])
		
		return participants


class RealTimeAPIService(APGBaseService):
	"""Real-time API service with WebSocket support."""
	
	def __init__(self):
		super().__init__()
		self.auth = APGAuth()
		self.security = APGSecurity()
		self.audit = APGAuditLogger()
		
		# Redis for pub/sub and event storage
		self.redis: Optional[aioredis.Redis] = None
		
		# WebSocket connections
		self.ws_connections: Dict[str, WebSocketConnection] = {}
		self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
		
		# Event management
		self.event_stream: Optional[EventStream] = None
		self.collaboration: Optional[RealTimeCollaboration] = None
		
		# Subscriptions
		self.subscriptions: Dict[str, EventSubscription] = {}
		self.subscription_index: Dict[str, Set[str]] = {}  # resource_id -> set of subscription_ids
		
		# SocketIO server for enhanced WebSocket support
		self.sio = socketio.AsyncServer(
			cors_allowed_origins="*",
			logger=False,
			engineio_logger=False
		)
		
		# Register SocketIO event handlers
		self._setup_socketio_handlers()
	
	async def start(self):
		"""Start the real-time API service."""
		await super().start()
		
		# Initialize Redis connection
		self.redis = aioredis.from_url("redis://localhost:6379", decode_responses=True)
		
		# Initialize event stream and collaboration
		self.event_stream = EventStream(self.redis)
		self.collaboration = RealTimeCollaboration(self.event_stream)
		
		# Start background tasks
		asyncio.create_task(self._connection_monitor())
		asyncio.create_task(self._event_subscriber())
		
		logger.info("Real-time API service started")
	
	async def stop(self):
		"""Stop the real-time API service."""
		# Close all WebSocket connections
		for connection in list(self.ws_connections.values()):
			await connection.ws.close()
		
		# Close Redis connection
		if self.redis:
			await self.redis.close()
		
		await super().stop()
		logger.info("Real-time API service stopped")
	
	def _setup_socketio_handlers(self):
		"""Setup SocketIO event handlers."""
		
		@self.sio.event
		async def connect(sid, environ, auth):
			"""Handle SocketIO connection."""
			try:
				# Authenticate user
				token = auth.get('token') if auth else None
				if not token:
					return False
				
				user = await self.auth.verify_token(token)
				if not user:
					return False
				
				# Store connection info
				await self.sio.save_session(sid, {
					'user_id': user.id,
					'tenant_id': user.tenant_id,
					'authenticated': True
				})
				
				logger.info(f"SocketIO connection established: {sid} for user {user.id}")
				return True
				
			except Exception as e:
				logger.error(f"SocketIO connection failed: {str(e)}")
				return False
		
		@self.sio.event
		async def disconnect(sid):
			"""Handle SocketIO disconnection."""
			try:
				session = await self.sio.get_session(sid)
				user_id = session.get('user_id')
				
				# Clean up subscriptions and sessions
				await self._cleanup_user_session(user_id, sid)
				
				logger.info(f"SocketIO disconnection: {sid}")
				
			except Exception as e:
				logger.error(f"SocketIO disconnect error: {str(e)}")
		
		@self.sio.event
		async def subscribe(sid, data):
			"""Handle event subscription."""
			try:
				session = await self.sio.get_session(sid)
				if not session.get('authenticated'):
					return {'error': 'Not authenticated'}
				
				subscription_id = await self._create_subscription(
					user_id=session['user_id'],
					session_id=sid,
					tenant_id=session['tenant_id'],
					subscription_data=data
				)
				
				return {'subscription_id': subscription_id}
				
			except Exception as e:
				logger.error(f"Subscription error: {str(e)}")
				return {'error': str(e)}
		
		@self.sio.event
		async def unsubscribe(sid, data):
			"""Handle event unsubscription."""
			try:
				subscription_id = data.get('subscription_id')
				if subscription_id:
					await self._remove_subscription(subscription_id)
				
				return {'success': True}
				
			except Exception as e:
				logger.error(f"Unsubscription error: {str(e)}")
				return {'error': str(e)}
		
		@self.sio.event
		async def join_collaboration(sid, data):
			"""Join collaboration session."""
			try:
				session = await self.sio.get_session(sid)
				if not session.get('authenticated'):
					return {'error': 'Not authenticated'}
				
				session_id = await self.collaboration.create_session(
					resource_type=data['resource_type'],
					resource_id=data['resource_id'],
					tenant_id=session['tenant_id'],
					user_id=session['user_id']
				)
				
				# Join SocketIO room
				await self.sio.enter_room(sid, session_id)
				
				return {
					'session_id': session_id,
					'participants': [asdict(p) for p in self.collaboration.get_session_participants(session_id)]
				}
				
			except Exception as e:
				logger.error(f"Collaboration join error: {str(e)}")
				return {'error': str(e)}
		
		@self.sio.event
		async def leave_collaboration(sid, data):
			"""Leave collaboration session."""
			try:
				session = await self.sio.get_session(sid)
				session_id = data.get('session_id')
				
				if session_id:
					await self.collaboration.leave_session(session_id, session['user_id'])
					await self.sio.leave_room(sid, session_id)
				
				return {'success': True}
				
			except Exception as e:
				logger.error(f"Collaboration leave error: {str(e)}")
				return {'error': str(e)}
		
		@self.sio.event
		async def cursor_update(sid, data):
			"""Update cursor position."""
			try:
				session = await self.sio.get_session(sid)
				session_id = data.get('session_id')
				cursor_data = data.get('cursor')
				
				if session_id and cursor_data:
					await self.collaboration.update_cursor(
						session_id, session['user_id'], cursor_data
					)
				
				return {'success': True}
				
			except Exception as e:
				logger.error(f"Cursor update error: {str(e)}")
				return {'error': str(e)}
	
	# WebSocket HTTP handler
	async def websocket_handler(self, request):
		"""Handle WebSocket connections via HTTP."""
		ws = web.WebSocketResponse()
		await ws.prepare(request)
		
		# Authenticate connection
		token = request.headers.get('Authorization', '').replace('Bearer ', '')
		if not token:
			await ws.close(code=4001, message='Authentication required')
			return ws
		
		user = await self.auth.verify_token(token)
		if not user:
			await ws.close(code=4001, message='Invalid token')
			return ws
		
		# Create connection
		session_id = uuid7str()
		connection = WebSocketConnection(ws, user.id, session_id, user.tenant_id)
		self.ws_connections[session_id] = connection
		
		if user.id not in self.user_connections:
			self.user_connections[user.id] = set()
		self.user_connections[user.id].add(session_id)
		
		try:
			async for msg in ws:
				if msg.type == WSMsgType.TEXT:
					try:
						data = json.loads(msg.data)
						await self._handle_ws_message(connection, data)
					except json.JSONDecodeError:
						await connection.send_message('error', {'message': 'Invalid JSON'})
				elif msg.type == WSMsgType.ERROR:
					logger.error(f'WebSocket error: {ws.exception()}')
		
		except Exception as e:
			logger.error(f"WebSocket handler error: {str(e)}")
		
		finally:
			# Clean up connection
			await self._cleanup_connection(connection)
		
		return ws
	
	async def _handle_ws_message(self, connection: WebSocketConnection, data: Dict[str, Any]):
		"""Handle WebSocket message."""
		message_type = data.get('type')
		
		if message_type == 'ping':
			connection.last_ping = datetime.utcnow()
			await connection.send_message('pong', {'timestamp': datetime.utcnow().isoformat()})
		
		elif message_type == 'subscribe':
			subscription_id = await self._create_subscription(
				user_id=connection.user_id,
				session_id=connection.session_id,
				tenant_id=connection.tenant_id,
				subscription_data=data.get('data', {})
			)
			connection.subscriptions.add(subscription_id)
			await connection.send_message('subscribed', {'subscription_id': subscription_id})
		
		elif message_type == 'unsubscribe':
			subscription_id = data.get('subscription_id')
			if subscription_id:
				await self._remove_subscription(subscription_id)
				connection.subscriptions.discard(subscription_id)
				await connection.send_message('unsubscribed', {'subscription_id': subscription_id})
		
		else:
			await connection.send_message('error', {'message': f'Unknown message type: {message_type}'})
	
	async def _create_subscription(self, user_id: str, session_id: str, tenant_id: str, 
								   subscription_data: Dict[str, Any]) -> str:
		"""Create event subscription."""
		subscription = EventSubscription(
			id=uuid7str(),
			user_id=user_id,
			session_id=session_id,
			tenant_id=tenant_id,
			subscription_type=SubscriptionType(subscription_data.get('type', 'workflow')),
			resource_id=subscription_data.get('resource_id'),
			event_types=[EventType(et) for et in subscription_data.get('event_types', [])],
			filters=subscription_data.get('filters', {})
		)
		
		self.subscriptions[subscription.id] = subscription
		
		# Index by resource for quick lookup
		index_key = f"{subscription.tenant_id}:{subscription.subscription_type}"
		if subscription.resource_id:
			index_key += f":{subscription.resource_id}"
		
		if index_key not in self.subscription_index:
			self.subscription_index[index_key] = set()
		self.subscription_index[index_key].add(subscription.id)
		
		return subscription.id
	
	async def _remove_subscription(self, subscription_id: str):
		"""Remove event subscription."""
		if subscription_id in self.subscriptions:
			subscription = self.subscriptions[subscription_id]
			
			# Remove from index
			index_key = f"{subscription.tenant_id}:{subscription.subscription_type}"
			if subscription.resource_id:
				index_key += f":{subscription.resource_id}"
			
			if index_key in self.subscription_index:
				self.subscription_index[index_key].discard(subscription_id)
				if not self.subscription_index[index_key]:
					del self.subscription_index[index_key]
			
			del self.subscriptions[subscription_id]
	
	async def _cleanup_connection(self, connection: WebSocketConnection):
		"""Clean up WebSocket connection."""
		# Remove subscriptions
		for subscription_id in connection.subscriptions:
			await self._remove_subscription(subscription_id)
		
		# Remove from connection tracking
		if connection.session_id in self.ws_connections:
			del self.ws_connections[connection.session_id]
		
		if connection.user_id in self.user_connections:
			self.user_connections[connection.user_id].discard(connection.session_id)
			if not self.user_connections[connection.user_id]:
				del self.user_connections[connection.user_id]
		
		# Leave collaboration sessions
		await self._cleanup_user_session(connection.user_id, connection.session_id)
	
	async def _cleanup_user_session(self, user_id: str, session_id: str):
		"""Clean up user session from collaboration."""
		# Find sessions the user was in and remove them
		sessions_to_leave = []
		for collab_session_id, session in self.collaboration.sessions.items():
			if user_id in session.participants:
				sessions_to_leave.append(collab_session_id)
		
		for collab_session_id in sessions_to_leave:
			await self.collaboration.leave_session(collab_session_id, user_id)
	
	async def _connection_monitor(self):
		"""Monitor WebSocket connections and clean up stale ones."""
		while True:
			try:
				current_time = datetime.utcnow()
				stale_connections = []
				
				for session_id, connection in self.ws_connections.items():
					# Check if connection is stale (no ping for 60 seconds)
					if (current_time - connection.last_ping).total_seconds() > 60:
						stale_connections.append(connection)
					# Check if WebSocket is closed
					elif not connection.is_alive():
						stale_connections.append(connection)
				
				# Clean up stale connections
				for connection in stale_connections:
					await self._cleanup_connection(connection)
					if not connection.ws.closed:
						await connection.ws.close()
				
				await asyncio.sleep(30)  # Check every 30 seconds
				
			except Exception as e:
				logger.error(f"Connection monitor error: {str(e)}")
				await asyncio.sleep(60)
	
	async def _event_subscriber(self):
		"""Subscribe to Redis events and distribute to WebSocket connections."""
		try:
			pubsub = self.redis.pubsub()
			await pubsub.psubscribe("realtime:*")
			
			async for message in pubsub.listen():
				if message['type'] == 'pmessage':
					try:
						channel = message['channel']
						tenant_id = channel.split(':')[1]
						event_data = json.loads(message['data'])
						
						# Create event object
						event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
						event = RealTimeEvent(**event_data)
						
						# Distribute to relevant connections
						await self._distribute_event(event)
						
					except Exception as e:
						logger.error(f"Event distribution error: {str(e)}")
		
		except Exception as e:
			logger.error(f"Event subscriber error: {str(e)}")
	
	async def _distribute_event(self, event: RealTimeEvent):
		"""Distribute event to relevant WebSocket connections."""
		# Find matching subscriptions
		matching_subscriptions = set()
		
		# Check tenant-wide subscriptions
		tenant_key = f"{event.tenant_id}:{event.resource_type}"
		if tenant_key in self.subscription_index:
			matching_subscriptions.update(self.subscription_index[tenant_key])
		
		# Check resource-specific subscriptions
		resource_key = f"{event.tenant_id}:{event.resource_type}:{event.resource_id}"
		if resource_key in self.subscription_index:
			matching_subscriptions.update(self.subscription_index[resource_key])
		
		# Send to matching connections
		for subscription_id in matching_subscriptions:
			if subscription_id in self.subscriptions:
				subscription = self.subscriptions[subscription_id]
				
				# Check event type filter
				if subscription.event_types and event.event_type not in subscription.event_types:
					continue
				
				# Check additional filters
				if subscription.filters:
					if not self._event_matches_filters(event, subscription.filters):
						continue
				
				# Send to connection
				if subscription.session_id in self.ws_connections:
					connection = self.ws_connections[subscription.session_id]
					await connection.send_event(event)
		
		# Also send via SocketIO
		await self._distribute_socketio_event(event)
	
	async def _distribute_socketio_event(self, event: RealTimeEvent):
		"""Distribute event via SocketIO."""
		try:
			# Send to tenant room
			room = f"tenant:{event.tenant_id}"
			await self.sio.emit('event', asdict(event), room=room)
			
			# Send to resource-specific room
			resource_room = f"{event.resource_type}:{event.resource_id}"
			await self.sio.emit('event', asdict(event), room=resource_room)
			
		except Exception as e:
			logger.error(f"SocketIO event distribution error: {str(e)}")
	
	def _event_matches_filters(self, event: RealTimeEvent, filters: Dict[str, Any]) -> bool:
		"""Check if event matches subscription filters."""
		for key, value in filters.items():
			if key == 'user_id' and event.user_id != value:
				return False
			elif key == 'source' and event.source != value:
				return False
			elif key in event.data and event.data[key] != value:
				return False
		
		return True
	
	# Public API methods
	
	async def publish_event(self, event: RealTimeEvent):
		"""Publish real-time event."""
		await self.event_stream.publish(event)
		
		# Audit the event
		await self.audit.log_event(
			AuditEvent(
				event_type='realtime_event_published',
				resource_type=event.resource_type,
				resource_id=event.resource_id,
				tenant_id=event.tenant_id,
				user_id=event.user_id,
				metadata={
					'event_type': event.event_type,
					'source': event.source
				}
			)
		)
	
	async def get_user_connections(self, user_id: str) -> List[WebSocketConnection]:
		"""Get all connections for a user."""
		connections = []
		if user_id in self.user_connections:
			for session_id in self.user_connections[user_id]:
				if session_id in self.ws_connections:
					connections.append(self.ws_connections[session_id])
		
		return connections
	
	async def send_to_user(self, user_id: str, message_type: str, data: Dict[str, Any]):
		"""Send message to all connections of a user."""
		connections = await self.get_user_connections(user_id)
		for connection in connections:
			await connection.send_message(message_type, data)
	
	async def get_collaboration_sessions(self, tenant_id: str) -> List[CollaborationSession]:
		"""Get all collaboration sessions for a tenant."""
		return [
			session for session in self.collaboration.sessions.values()
			if session.tenant_id == tenant_id
		]
	
	async def get_event_history(self, tenant_id: str, resource_type: str, 
								resource_id: str, limit: int = 50) -> List[RealTimeEvent]:
		"""Get event history for a resource."""
		return await self.event_stream.get_event_history(
			tenant_id, resource_type, resource_id, limit
		)


# Global service instance
realtime_api_service = RealTimeAPIService()
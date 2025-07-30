"""
APG Notification Capability - WebSocket Real-Time Events

Real-time collaboration and event streaming for notification management
with live campaign monitoring, template editing, and system status updates.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import jwt
import redis
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_socketio import Namespace

from .api_models import (
	DeliveryChannel, NotificationPriority, EngagementEvent,
	ComprehensiveDelivery, AdvancedCampaign
)


# Configure logging
_log = logging.getLogger(__name__)


class EventType(str, Enum):
	"""WebSocket event types"""
	# Real-time delivery events
	DELIVERY_STARTED = "delivery.started"
	DELIVERY_PROGRESS = "delivery.progress"
	DELIVERY_COMPLETED = "delivery.completed"
	DELIVERY_FAILED = "delivery.failed"
	DELIVERY_ENGAGEMENT = "delivery.engagement"
	
	# Campaign events
	CAMPAIGN_STARTED = "campaign.started"
	CAMPAIGN_PROGRESS = "campaign.progress"
	CAMPAIGN_COMPLETED = "campaign.completed"
	CAMPAIGN_PAUSED = "campaign.paused"
	CAMPAIGN_RESUMED = "campaign.resumed"
	
	# Real-time collaboration events
	CAMPAIGN_EDITING_STARTED = "campaign.editing.started"
	CAMPAIGN_EDITING_STOPPED = "campaign.editing.stopped"
	CAMPAIGN_CONTENT_CHANGED = "campaign.content.changed"
	CAMPAIGN_USER_JOINED = "campaign.user.joined"
	CAMPAIGN_USER_LEFT = "campaign.user.left"
	CAMPAIGN_CURSOR_MOVED = "campaign.cursor.moved"
	CAMPAIGN_COMMENT_ADDED = "campaign.comment.added"
	
	# Template collaboration events
	TEMPLATE_EDITING_STARTED = "template.editing.started"
	TEMPLATE_EDITING_STOPPED = "template.editing.stopped"
	TEMPLATE_CONTENT_CHANGED = "template.content.changed"
	TEMPLATE_USER_JOINED = "template.user.joined"
	TEMPLATE_USER_LEFT = "template.user.left"
	
	# System and monitoring events
	SYSTEM_HEALTH_UPDATE = "system.health.update"
	CHANNEL_STATUS_CHANGE = "channel.status.change"
	PERFORMANCE_ALERT = "performance.alert"
	QUEUE_SIZE_UPDATE = "queue.size.update"
	
	# Analytics events
	ANALYTICS_UPDATE = "analytics.update"
	ENGAGEMENT_SPIKE = "engagement.spike"
	CONVERSION_MILESTONE = "conversion.milestone"
	
	# User events
	USER_PREFERENCES_UPDATED = "user.preferences.updated"
	USER_FEEDBACK_RECEIVED = "user.feedback.received"


@dataclass
class WebSocketEvent:
	"""WebSocket event data structure"""
	event_type: EventType
	data: Dict[str, Any]
	timestamp: datetime
	tenant_id: str
	user_id: Optional[str] = None
	session_id: Optional[str] = None
	room: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert event to dictionary"""
		return {
			'event_type': self.event_type.value,
			'data': self.data,
			'timestamp': self.timestamp.isoformat(),
			'tenant_id': self.tenant_id,
			'user_id': self.user_id,
			'session_id': self.session_id,
			'room': self.room
		}


@dataclass
class CollaborativeSession:
	"""Real-time collaborative editing session"""
	session_id: str
	resource_type: str  # 'campaign' or 'template'
	resource_id: str
	tenant_id: str
	participants: Set[str]  # user_ids
	created_at: datetime
	last_activity: datetime
	
	def add_participant(self, user_id: str):
		"""Add participant to session"""
		self.participants.add(user_id)
		self.last_activity = datetime.utcnow()
	
	def remove_participant(self, user_id: str):
		"""Remove participant from session"""
		self.participants.discard(user_id)
		self.last_activity = datetime.utcnow()
	
	def is_active(self, timeout_minutes: int = 30) -> bool:
		"""Check if session is still active"""
		timeout = datetime.utcnow().timestamp() - (timeout_minutes * 60)
		return self.last_activity.timestamp() > timeout


class NotificationWebSocketManager:
	"""
	WebSocket manager for real-time notification events and collaboration.
	Handles event broadcasting, room management, and collaborative editing.
	"""
	
	def __init__(self, app: Flask, redis_url: str = "redis://localhost:6379"):
		"""Initialize WebSocket manager"""
		self.app = app
		self.socketio = SocketIO(
			app,
			cors_allowed_origins="*",  # Configure for production
			async_mode='threading',
			logger=True,
			engineio_logger=True,
			message_queue=redis_url
		)
		
		# Redis for session management and event broadcasting
		self.redis_client = redis.from_url(redis_url, decode_responses=True)
		
		# Active collaborative sessions
		self.collaborative_sessions: Dict[str, CollaborativeSession] = {}
		
		# Connected clients by tenant
		self.connected_clients: Dict[str, Set[str]] = {}  # tenant_id -> session_ids
		
		# Event handlers registry
		self.event_handlers: Dict[EventType, List[Callable]] = {}
		
		# Register namespaces
		self._register_namespaces()
		
		_log.info("NotificationWebSocketManager initialized")
	
	def _register_namespaces(self):
		"""Register WebSocket namespaces"""
		
		# Real-time monitoring namespace
		monitoring_ns = MonitoringNamespace('/monitoring')
		self.socketio.on_namespace(monitoring_ns)
		
		# Collaboration namespace  
		collaboration_ns = CollaborationNamespace('/collaboration')
		self.socketio.on_namespace(collaboration_ns)
		
		# Analytics namespace
		analytics_ns = AnalyticsNamespace('/analytics')
		self.socketio.on_namespace(analytics_ns)
		
		# Set manager reference in namespaces
		monitoring_ns.manager = self
		collaboration_ns.manager = self
		analytics_ns.manager = self
	
	# ========== Event Broadcasting ==========
	
	async def broadcast_event(
		self,
		event: WebSocketEvent,
		room: Optional[str] = None
	):
		"""Broadcast event to clients"""
		try:
			# Determine target room
			target_room = room or event.room or f"tenant_{event.tenant_id}"
			
			# Emit event to room
			self.socketio.emit(
				event.event_type.value,
				event.to_dict(),
				room=target_room,
				namespace='/'
			)
			
			# Store event in Redis for replay
			await self._store_event(event)
			
			# Trigger registered handlers
			await self._trigger_event_handlers(event)
			
			_log.debug(f"Broadcasted event {event.event_type.value} to room {target_room}")
			
		except Exception as e:
			_log.error(f"Failed to broadcast event: {str(e)}")
	
	async def broadcast_delivery_event(
		self,
		delivery: ComprehensiveDelivery,
		event_type: EventType
	):
		"""Broadcast delivery-related event"""
		event = WebSocketEvent(
			event_type=event_type,
			data={
				'delivery_id': delivery.id,
				'campaign_id': delivery.campaign_id,
				'recipient_id': delivery.recipient_id,
				'channels': [c.value for c in delivery.channels],
				'status': delivery.status,
				'latency_ms': delivery.delivery_latency_ms,
				'cost': delivery.cost
			},
			timestamp=datetime.utcnow(),
			tenant_id=delivery.tenant_id,
			room=f"delivery_{delivery.id}"
		)
		
		await self.broadcast_event(event)
	
	async def broadcast_campaign_event(
		self,
		campaign: AdvancedCampaign,
		event_type: EventType,
		additional_data: Optional[Dict[str, Any]] = None
	):
		"""Broadcast campaign-related event"""
		data = {
			'campaign_id': campaign.id,
			'name': campaign.name,
			'campaign_type': campaign.campaign_type.value,
			'status': campaign.status,
			'total_recipients': campaign.total_recipients,
			'execution_count': campaign.execution_count
		}
		
		if additional_data:
			data.update(additional_data)
		
		event = WebSocketEvent(
			event_type=event_type,
			data=data,
			timestamp=datetime.utcnow(),
			tenant_id=campaign.tenant_id,
			room=f"campaign_{campaign.id}"
		)
		
		await self.broadcast_event(event)
	
	async def broadcast_system_health(
		self,
		health_data: Dict[str, Any],
		tenant_id: str
	):
		"""Broadcast system health update"""
		event = WebSocketEvent(
			event_type=EventType.SYSTEM_HEALTH_UPDATE,
			data=health_data,
			timestamp=datetime.utcnow(),
			tenant_id=tenant_id,
			room=f"tenant_{tenant_id}_system"
		)
		
		await self.broadcast_event(event)
	
	# ========== Collaborative Editing ==========
	
	async def start_collaborative_session(
		self,
		resource_type: str,
		resource_id: str,
		tenant_id: str,
		user_id: str
	) -> str:
		"""Start collaborative editing session"""
		session_id = f"{resource_type}_{resource_id}_{datetime.utcnow().timestamp()}"
		
		session = CollaborativeSession(
			session_id=session_id,
			resource_type=resource_type,
			resource_id=resource_id,
			tenant_id=tenant_id,
			participants={user_id},
			created_at=datetime.utcnow(),
			last_activity=datetime.utcnow()
		)
		
		self.collaborative_sessions[session_id] = session
		
		# Broadcast session start
		event = WebSocketEvent(
			event_type=EventType.CAMPAIGN_EDITING_STARTED if resource_type == 'campaign' else EventType.TEMPLATE_EDITING_STARTED,
			data={
				'session_id': session_id,
				'resource_type': resource_type,
				'resource_id': resource_id,
				'participants': list(session.participants)
			},
			timestamp=datetime.utcnow(),
			tenant_id=tenant_id,
			user_id=user_id,
			room=f"{resource_type}_{resource_id}"
		)
		
		await self.broadcast_event(event)
		
		_log.info(f"Started collaborative session {session_id}")
		return session_id
	
	async def join_collaborative_session(
		self,
		session_id: str,
		user_id: str
	):
		"""Join collaborative editing session"""
		if session_id not in self.collaborative_sessions:
			raise ValueError(f"Session {session_id} not found")
		
		session = self.collaborative_sessions[session_id]
		session.add_participant(user_id)
		
		# Broadcast user joined
		event = WebSocketEvent(
			event_type=EventType.CAMPAIGN_USER_JOINED if session.resource_type == 'campaign' else EventType.TEMPLATE_USER_JOINED,
			data={
				'session_id': session_id,
				'user_id': user_id,
				'participants': list(session.participants)
			},
			timestamp=datetime.utcnow(),
			tenant_id=session.tenant_id,
			user_id=user_id,
			room=f"{session.resource_type}_{session.resource_id}"
		)
		
		await self.broadcast_event(event)
	
	async def leave_collaborative_session(
		self,
		session_id: str,
		user_id: str
	):
		"""Leave collaborative editing session"""
		if session_id not in self.collaborative_sessions:
			return
		
		session = self.collaborative_sessions[session_id]
		session.remove_participant(user_id)
		
		# Broadcast user left
		event = WebSocketEvent(
			event_type=EventType.CAMPAIGN_USER_LEFT if session.resource_type == 'campaign' else EventType.TEMPLATE_USER_LEFT,
			data={
				'session_id': session_id,
				'user_id': user_id,
				'participants': list(session.participants)
			},
			timestamp=datetime.utcnow(),
			tenant_id=session.tenant_id,
			user_id=user_id,
			room=f"{session.resource_type}_{session.resource_id}"
		)
		
		await self.broadcast_event(event)
		
		# Clean up empty session
		if not session.participants:
			del self.collaborative_sessions[session_id]
			_log.info(f"Cleaned up empty session {session_id}")
	
	async def broadcast_content_change(
		self,
		session_id: str,
		user_id: str,
		changes: Dict[str, Any]
	):
		"""Broadcast content changes in collaborative session"""
		if session_id not in self.collaborative_sessions:
			return
		
		session = self.collaborative_sessions[session_id]
		session.last_activity = datetime.utcnow()
		
		event = WebSocketEvent(
			event_type=EventType.CAMPAIGN_CONTENT_CHANGED if session.resource_type == 'campaign' else EventType.TEMPLATE_CONTENT_CHANGED,
			data={
				'session_id': session_id,
				'user_id': user_id,
				'changes': changes,
				'timestamp': datetime.utcnow().isoformat()
			},
			timestamp=datetime.utcnow(),
			tenant_id=session.tenant_id,
			user_id=user_id,
			room=f"{session.resource_type}_{session.resource_id}"
		)
		
		await self.broadcast_event(event)
	
	# ========== Event Management ==========
	
	def register_event_handler(
		self,
		event_type: EventType,
		handler: Callable[[WebSocketEvent], None]
	):
		"""Register event handler"""
		if event_type not in self.event_handlers:
			self.event_handlers[event_type] = []
		
		self.event_handlers[event_type].append(handler)
		_log.info(f"Registered handler for {event_type.value}")
	
	async def _trigger_event_handlers(self, event: WebSocketEvent):
		"""Trigger registered event handlers"""
		handlers = self.event_handlers.get(event.event_type, [])
		
		for handler in handlers:
			try:
				if asyncio.iscoroutinefunction(handler):
					await handler(event)
				else:
					handler(event)
			except Exception as e:
				_log.error(f"Event handler failed for {event.event_type.value}: {str(e)}")
	
	async def _store_event(self, event: WebSocketEvent):
		"""Store event in Redis for replay and persistence"""
		try:
			key = f"events:{event.tenant_id}:{event.event_type.value}"
			event_data = json.dumps(event.to_dict(), default=str)
			
			# Store with expiration (24 hours)
			self.redis_client.lpush(key, event_data)
			self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 events
			self.redis_client.expire(key, 86400)  # 24 hours
			
		except Exception as e:
			_log.error(f"Failed to store event in Redis: {str(e)}")
	
	# ========== Session Management ==========
	
	def cleanup_inactive_sessions(self, timeout_minutes: int = 30):
		"""Clean up inactive collaborative sessions"""
		inactive_sessions = []
		
		for session_id, session in self.collaborative_sessions.items():
			if not session.is_active(timeout_minutes):
				inactive_sessions.append(session_id)
		
		for session_id in inactive_sessions:
			del self.collaborative_sessions[session_id]
			_log.info(f"Cleaned up inactive session {session_id}")
	
	def get_session_stats(self) -> Dict[str, Any]:
		"""Get session statistics"""
		return {
			'active_sessions': len(self.collaborative_sessions),
			'total_participants': sum(
				len(session.participants) 
				for session in self.collaborative_sessions.values()
			),
			'sessions_by_type': {
				'campaign': len([
					s for s in self.collaborative_sessions.values() 
					if s.resource_type == 'campaign'
				]),
				'template': len([
					s for s in self.collaborative_sessions.values() 
					if s.resource_type == 'template'
				])
			},
			'connected_tenants': len(self.connected_clients)
		}


# ========== WebSocket Namespaces ==========

class MonitoringNamespace(Namespace):
	"""Real-time monitoring namespace"""
	
	def __init__(self, namespace):
		super().__init__(namespace)
		self.manager = None
	
	def on_connect(self, auth):
		"""Handle client connection"""
		try:
			# Validate authentication
			if not auth or 'token' not in auth:
				disconnect()
				return
			
			# Decode JWT token (simplified)
			# token_data = jwt.decode(auth['token'], verify=False)
			# tenant_id = token_data.get('tenant_id')
			# user_id = token_data.get('user_id')
			
			# Mock authentication for demo
			tenant_id = auth.get('tenant_id', 'default_tenant')
			user_id = auth.get('user_id', 'user_123')
			
			# Join tenant room
			join_room(f"tenant_{tenant_id}")
			join_room(f"tenant_{tenant_id}_system")
			
			_log.info(f"Client connected to monitoring: user {user_id}, tenant {tenant_id}")
			
			# Send current system status
			emit('system_status', {
				'status': 'connected',
				'tenant_id': tenant_id,
				'timestamp': datetime.utcnow().isoformat()
			})
			
		except Exception as e:
			_log.error(f"Connection failed: {str(e)}")
			disconnect()
	
	def on_disconnect(self):
		"""Handle client disconnection"""
		_log.info("Client disconnected from monitoring")
	
	def on_subscribe_delivery(self, data):
		"""Subscribe to delivery updates"""
		delivery_id = data.get('delivery_id')
		if delivery_id:
			join_room(f"delivery_{delivery_id}")
			emit('subscribed', {'delivery_id': delivery_id})
	
	def on_subscribe_campaign(self, data):
		"""Subscribe to campaign updates"""
		campaign_id = data.get('campaign_id')
		if campaign_id:
			join_room(f"campaign_{campaign_id}")
			emit('subscribed', {'campaign_id': campaign_id})


class CollaborationNamespace(Namespace):
	"""Real-time collaboration namespace"""
	
	def __init__(self, namespace):
		super().__init__(namespace)
		self.manager = None
	
	def on_connect(self, auth):
		"""Handle collaboration connection"""
		try:
			# Validate and extract user info
			tenant_id = auth.get('tenant_id', 'default_tenant')
			user_id = auth.get('user_id', 'user_123')
			
			_log.info(f"Client connected to collaboration: user {user_id}, tenant {tenant_id}")
			
			emit('collaboration_ready', {
				'status': 'connected',
				'user_id': user_id,
				'tenant_id': tenant_id
			})
			
		except Exception as e:
			_log.error(f"Collaboration connection failed: {str(e)}")
			disconnect()
	
	def on_start_editing(self, data):
		"""Start collaborative editing session"""
		try:
			resource_type = data.get('resource_type')  # 'campaign' or 'template'
			resource_id = data.get('resource_id')
			tenant_id = data.get('tenant_id', 'default_tenant')
			user_id = data.get('user_id', 'user_123')
			
			# Start session
			if self.manager:
				session_id = asyncio.create_task(
					self.manager.start_collaborative_session(
						resource_type, resource_id, tenant_id, user_id
					)
				)
				
				# Join resource room
				join_room(f"{resource_type}_{resource_id}")
				
				emit('editing_started', {
					'session_id': session_id,
					'resource_type': resource_type,
					'resource_id': resource_id
				})
			
		except Exception as e:
			_log.error(f"Start editing failed: {str(e)}")
			emit('error', {'message': str(e)})
	
	def on_content_change(self, data):
		"""Handle content changes"""
		try:
			session_id = data.get('session_id')
			user_id = data.get('user_id', 'user_123')
			changes = data.get('changes', {})
			
			if self.manager:
				asyncio.create_task(
					self.manager.broadcast_content_change(
						session_id, user_id, changes
					)
				)
			
		except Exception as e:
			_log.error(f"Content change failed: {str(e)}")
			emit('error', {'message': str(e)})
	
	def on_cursor_move(self, data):
		"""Handle cursor movement"""
		try:
			resource_id = data.get('resource_id')
			resource_type = data.get('resource_type')
			user_id = data.get('user_id', 'user_123')
			position = data.get('position', {})
			
			# Broadcast cursor position to other users in room
			emit('cursor_moved', {
				'user_id': user_id,
				'position': position,
				'timestamp': datetime.utcnow().isoformat()
			}, room=f"{resource_type}_{resource_id}")
			
		except Exception as e:
			_log.error(f"Cursor move failed: {str(e)}")


class AnalyticsNamespace(Namespace):
	"""Real-time analytics namespace"""
	
	def __init__(self, namespace):
		super().__init__(namespace)
		self.manager = None
	
	def on_connect(self, auth):
		"""Handle analytics connection"""
		try:
			tenant_id = auth.get('tenant_id', 'default_tenant')
			user_id = auth.get('user_id', 'user_123')
			
			# Join analytics room
			join_room(f"analytics_{tenant_id}")
			
			_log.info(f"Client connected to analytics: user {user_id}, tenant {tenant_id}")
			
			emit('analytics_ready', {
				'status': 'connected',
				'tenant_id': tenant_id
			})
			
		except Exception as e:
			_log.error(f"Analytics connection failed: {str(e)}")
			disconnect()
	
	def on_subscribe_metrics(self, data):
		"""Subscribe to real-time metrics"""
		metric_types = data.get('metrics', [])
		tenant_id = data.get('tenant_id', 'default_tenant')
		
		for metric_type in metric_types:
			join_room(f"metrics_{tenant_id}_{metric_type}")
		
		emit('metrics_subscribed', {'metrics': metric_types})


def create_websocket_manager(app: Flask, redis_url: str = "redis://localhost:6379") -> NotificationWebSocketManager:
	"""Create and configure WebSocket manager"""
	return NotificationWebSocketManager(app, redis_url)


# Export main classes and functions
__all__ = [
	'NotificationWebSocketManager',
	'EventType',
	'WebSocketEvent',
	'CollaborativeSession',
	'MonitoringNamespace',
	'CollaborationNamespace', 
	'AnalyticsNamespace',
	'create_websocket_manager'
]
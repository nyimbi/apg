"""
Time & Attendance Capability WebSocket Manager

Real-time communication for live dashboards, instant notifications,
and collaborative features in the revolutionary APG Time & Attendance capability.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from uuid import uuid4
from dataclasses import dataclass, asdict

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .config import get_config
from .models import TimeEntryStatus, RemoteWorkStatus, AIAgentType


logger = logging.getLogger(__name__)


@dataclass
class WebSocketClient:
	"""WebSocket client information"""
	websocket: WebSocket
	client_id: str
	user_id: str
	tenant_id: str
	subscriptions: Set[str]
	connected_at: datetime
	last_ping: datetime


class WebSocketMessage(BaseModel):
	"""WebSocket message model"""
	type: str = Field(..., description="Message type")
	channel: str = Field(..., description="Channel/topic")
	data: Dict[str, Any] = Field(..., description="Message payload")
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	client_id: Optional[str] = Field(None, description="Source client ID")


class RealTimeEvent(BaseModel):
	"""Real-time event model"""
	event_type: str = Field(..., description="Event type")
	entity_type: str = Field(..., description="Entity type (employee, time_entry, etc.)")
	entity_id: str = Field(..., description="Entity identifier")
	tenant_id: str = Field(..., description="Tenant identifier")
	data: Dict[str, Any] = Field(..., description="Event data")
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	user_id: Optional[str] = Field(None, description="User who triggered event")


class WebSocketManager:
	"""
	Comprehensive WebSocket manager for real-time features
	
	Handles live dashboards, instant notifications, collaborative editing,
	and real-time workforce analytics.
	"""
	
	def __init__(self):
		self.config = get_config()
		self.clients: Dict[str, WebSocketClient] = {}
		self.tenant_clients: Dict[str, Set[str]] = {}  # tenant_id -> client_ids
		self.user_clients: Dict[str, Set[str]] = {}    # user_id -> client_ids
		self.channel_subscribers: Dict[str, Set[str]] = {}  # channel -> client_ids
		self.heartbeat_interval = 30  # seconds
		self.cleanup_interval = 300   # seconds
		self._running = False
		
		logger.info("WebSocket manager initialized")
	
	async def start_background_tasks(self):
		"""Start background maintenance tasks"""
		self._running = True
		asyncio.create_task(self._heartbeat_task())
		asyncio.create_task(self._cleanup_task())
		logger.info("WebSocket background tasks started")
	
	async def stop_background_tasks(self):
		"""Stop background tasks"""
		self._running = False
		logger.info("WebSocket background tasks stopped")
	
	async def connect(
		self, 
		websocket: WebSocket, 
		user_id: str, 
		tenant_id: str,
		client_id: Optional[str] = None
	) -> str:
		"""Connect a new WebSocket client"""
		await websocket.accept()
		
		# Generate client ID if not provided
		if not client_id:
			client_id = str(uuid4())
		
		# Create client record
		client = WebSocketClient(
			websocket=websocket,
			client_id=client_id,
			user_id=user_id,
			tenant_id=tenant_id,
			subscriptions=set(),
			connected_at=datetime.utcnow(),
			last_ping=datetime.utcnow()
		)
		
		# Store client
		self.clients[client_id] = client
		
		# Index by tenant
		if tenant_id not in self.tenant_clients:
			self.tenant_clients[tenant_id] = set()
		self.tenant_clients[tenant_id].add(client_id)
		
		# Index by user
		if user_id not in self.user_clients:
			self.user_clients[user_id] = set()
		self.user_clients[user_id].add(client_id)
		
		logger.info(f"WebSocket client connected: {client_id} (user: {user_id}, tenant: {tenant_id})")
		
		# Send welcome message
		await self._send_to_client(client_id, WebSocketMessage(
			type="connection",
			channel="system",
			data={
				"status": "connected",
				"client_id": client_id,
				"server_time": datetime.utcnow().isoformat(),
				"features": {
					"real_time_dashboard": True,
					"instant_notifications": True,
					"collaborative_editing": True,
					"live_analytics": True
				}
			}
		))
		
		return client_id
	
	async def disconnect(self, client_id: str):
		"""Disconnect a WebSocket client"""
		if client_id not in self.clients:
			return
		
		client = self.clients[client_id]
		
		# Remove from indexes
		if client.tenant_id in self.tenant_clients:
			self.tenant_clients[client.tenant_id].discard(client_id)
			if not self.tenant_clients[client.tenant_id]:
				del self.tenant_clients[client.tenant_id]
		
		if client.user_id in self.user_clients:
			self.user_clients[client.user_id].discard(client_id)
			if not self.user_clients[client.user_id]:
				del self.user_clients[client.user_id]
		
		# Remove from channel subscriptions
		for channel in client.subscriptions:
			if channel in self.channel_subscribers:
				self.channel_subscribers[channel].discard(client_id)
				if not self.channel_subscribers[channel]:
					del self.channel_subscribers[channel]
		
		# Remove client
		del self.clients[client_id]
		
		logger.info(f"WebSocket client disconnected: {client_id}")
	
	async def subscribe(self, client_id: str, channel: str):
		"""Subscribe client to a channel"""
		if client_id not in self.clients:
			return False
		
		client = self.clients[client_id]
		client.subscriptions.add(channel)
		
		if channel not in self.channel_subscribers:
			self.channel_subscribers[channel] = set()
		self.channel_subscribers[channel].add(client_id)
		
		logger.debug(f"Client {client_id} subscribed to channel: {channel}")
		
		# Send subscription confirmation
		await self._send_to_client(client_id, WebSocketMessage(
			type="subscription",
			channel="system",
			data={
				"action": "subscribed",
				"channel": channel,
				"status": "success"
			}
		))
		
		return True
	
	async def unsubscribe(self, client_id: str, channel: str):
		"""Unsubscribe client from a channel"""
		if client_id not in self.clients:
			return False
		
		client = self.clients[client_id]
		client.subscriptions.discard(channel)
		
		if channel in self.channel_subscribers:
			self.channel_subscribers[channel].discard(client_id)
			if not self.channel_subscribers[channel]:
				del self.channel_subscribers[channel]
		
		logger.debug(f"Client {client_id} unsubscribed from channel: {channel}")
		return True
	
	async def broadcast_to_tenant(
		self, 
		tenant_id: str, 
		message: WebSocketMessage,
		exclude_client: Optional[str] = None
	):
		"""Broadcast message to all clients in a tenant"""
		if tenant_id not in self.tenant_clients:
			return
		
		client_ids = self.tenant_clients[tenant_id].copy()
		if exclude_client:
			client_ids.discard(exclude_client)
		
		await self._send_to_clients(client_ids, message)
		logger.debug(f"Broadcasted to tenant {tenant_id}: {len(client_ids)} clients")
	
	async def broadcast_to_channel(
		self, 
		channel: str, 
		message: WebSocketMessage,
		exclude_client: Optional[str] = None
	):
		"""Broadcast message to all subscribers of a channel"""
		if channel not in self.channel_subscribers:
			return
		
		client_ids = self.channel_subscribers[channel].copy()
		if exclude_client:
			client_ids.discard(exclude_client)
		
		await self._send_to_clients(client_ids, message)
		logger.debug(f"Broadcasted to channel {channel}: {len(client_ids)} clients")
	
	async def send_to_user(
		self, 
		user_id: str, 
		message: WebSocketMessage
	):
		"""Send message to all connections of a specific user"""
		if user_id not in self.user_clients:
			return
		
		client_ids = self.user_clients[user_id].copy()
		await self._send_to_clients(client_ids, message)
		logger.debug(f"Sent to user {user_id}: {len(client_ids)} clients")
	
	async def handle_client_message(self, client_id: str, message: Dict[str, Any]):
		"""Handle incoming message from client"""
		if client_id not in self.clients:
			return
		
		client = self.clients[client_id]
		client.last_ping = datetime.utcnow()
		
		try:
			msg_type = message.get("type")
			
			if msg_type == "ping":
				await self._handle_ping(client_id)
			elif msg_type == "subscribe":
				await self._handle_subscribe(client_id, message)
			elif msg_type == "unsubscribe":
				await self._handle_unsubscribe(client_id, message)
			elif msg_type == "dashboard_request":
				await self._handle_dashboard_request(client_id, message)
			elif msg_type == "live_edit":
				await self._handle_live_edit(client_id, message)
			else:
				logger.warning(f"Unknown message type from client {client_id}: {msg_type}")
		
		except Exception as e:
			logger.error(f"Error handling message from client {client_id}: {str(e)}")
			await self._send_error(client_id, f"Message handling error: {str(e)}")
	
	# Real-time event broadcasting methods
	
	async def broadcast_time_entry_event(self, event: RealTimeEvent):
		"""Broadcast time entry related events"""
		message = WebSocketMessage(
			type="time_entry_event",
			channel="time_entries",
			data={
				"event_type": event.event_type,
				"entity_id": event.entity_id,
				"entity_data": event.data,
				"user_id": event.user_id,
				"timestamp": event.timestamp.isoformat()
			}
		)
		
		# Broadcast to tenant
		await self.broadcast_to_tenant(event.tenant_id, message)
		
		# Broadcast to specific channels
		await self.broadcast_to_channel(f"employee_{event.data.get('employee_id')}", message)
		await self.broadcast_to_channel("live_dashboard", message)
	
	async def broadcast_remote_work_event(self, event: RealTimeEvent):
		"""Broadcast remote work related events"""
		message = WebSocketMessage(
			type="remote_work_event",
			channel="remote_work",
			data={
				"event_type": event.event_type,
				"entity_id": event.entity_id,
				"entity_data": event.data,
				"timestamp": event.timestamp.isoformat()
			}
		)
		
		await self.broadcast_to_tenant(event.tenant_id, message)
		await self.broadcast_to_channel("remote_dashboard", message)
	
	async def broadcast_ai_agent_event(self, event: RealTimeEvent):
		"""Broadcast AI agent related events"""
		message = WebSocketMessage(
			type="ai_agent_event",
			channel="ai_agents",
			data={
				"event_type": event.event_type,
				"entity_id": event.entity_id,
				"entity_data": event.data,
				"timestamp": event.timestamp.isoformat()
			}
		)
		
		await self.broadcast_to_tenant(event.tenant_id, message)
		await self.broadcast_to_channel("ai_dashboard", message)
	
	async def broadcast_collaboration_event(self, event: RealTimeEvent):
		"""Broadcast hybrid collaboration events"""
		message = WebSocketMessage(
			type="collaboration_event",
			channel="collaboration",
			data={
				"event_type": event.event_type,
				"entity_id": event.entity_id,
				"entity_data": event.data,
				"timestamp": event.timestamp.isoformat()
			}
		)
		
		await self.broadcast_to_tenant(event.tenant_id, message)
		
		# Notify specific collaboration participants
		if "human_participants" in event.data:
			for participant_id in event.data["human_participants"]:
				await self.send_to_user(participant_id, message)
	
	async def broadcast_fraud_alert(self, event: RealTimeEvent):
		"""Broadcast fraud detection alerts"""
		message = WebSocketMessage(
			type="fraud_alert",
			channel="security",
			data={
				"alert_type": "fraud_detection",
				"severity": event.data.get("severity", "medium"),
				"entity_id": event.entity_id,
				"details": event.data,
				"timestamp": event.timestamp.isoformat()
			}
		)
		
		# Send to managers and security personnel
		await self.broadcast_to_channel("security_alerts", message)
		await self.broadcast_to_channel("manager_dashboard", message)
	
	# Private helper methods
	
	async def _send_to_client(self, client_id: str, message: WebSocketMessage):
		"""Send message to a specific client"""
		if client_id not in self.clients:
			return
		
		client = self.clients[client_id]
		try:
			await client.websocket.send_text(json.dumps(message.dict(), default=str))
		except Exception as e:
			logger.error(f"Error sending to client {client_id}: {str(e)}")
			await self.disconnect(client_id)
	
	async def _send_to_clients(self, client_ids: Set[str], message: WebSocketMessage):
		"""Send message to multiple clients"""
		message_json = json.dumps(message.dict(), default=str)
		
		# Send concurrently to all clients
		tasks = []
		for client_id in client_ids:
			if client_id in self.clients:
				client = self.clients[client_id]
				tasks.append(self._safe_send(client, message_json))
		
		if tasks:
			await asyncio.gather(*tasks, return_exceptions=True)
	
	async def _safe_send(self, client: WebSocketClient, message_json: str):
		"""Safely send message to client with error handling"""
		try:
			await client.websocket.send_text(message_json)
		except Exception as e:
			logger.error(f"Error sending to client {client.client_id}: {str(e)}")
			await self.disconnect(client.client_id)
	
	async def _send_error(self, client_id: str, error_message: str):
		"""Send error message to client"""
		error_msg = WebSocketMessage(
			type="error",
			channel="system",
			data={
				"error": error_message,
				"timestamp": datetime.utcnow().isoformat()
			}
		)
		await self._send_to_client(client_id, error_msg)
	
	async def _handle_ping(self, client_id: str):
		"""Handle ping message"""
		pong_msg = WebSocketMessage(
			type="pong",
			channel="system",
			data={"timestamp": datetime.utcnow().isoformat()}
		)
		await self._send_to_client(client_id, pong_msg)
	
	async def _handle_subscribe(self, client_id: str, message: Dict[str, Any]):
		"""Handle subscription request"""
		channel = message.get("channel")
		if channel:
			await self.subscribe(client_id, channel)
	
	async def _handle_unsubscribe(self, client_id: str, message: Dict[str, Any]):
		"""Handle unsubscription request"""
		channel = message.get("channel")
		if channel:
			await self.unsubscribe(client_id, channel)
	
	async def _handle_dashboard_request(self, client_id: str, message: Dict[str, Any]):
		"""Handle dashboard data request"""
		dashboard_type = message.get("dashboard_type", "overview")
		
		# Get client info
		client = self.clients[client_id]
		
		# Generate dashboard data based on type
		dashboard_data = await self._generate_dashboard_data(
			dashboard_type, client.tenant_id, client.user_id
		)
		
		response = WebSocketMessage(
			type="dashboard_data",
			channel="dashboard",
			data={
				"dashboard_type": dashboard_type,
				"data": dashboard_data,
				"timestamp": datetime.utcnow().isoformat()
			}
		)
		
		await self._send_to_client(client_id, response)
	
	async def _handle_live_edit(self, client_id: str, message: Dict[str, Any]):
		"""Handle live editing collaboration"""
		entity_type = message.get("entity_type")
		entity_id = message.get("entity_id")
		changes = message.get("changes")
		
		if not all([entity_type, entity_id, changes]):
			await self._send_error(client_id, "Missing required fields for live edit")
			return
		
		# Broadcast changes to other collaborators
		client = self.clients[client_id]
		
		edit_message = WebSocketMessage(
			type="live_edit",
			channel=f"{entity_type}_{entity_id}",
			data={
				"entity_type": entity_type,
				"entity_id": entity_id,
				"changes": changes,
				"user_id": client.user_id,
				"timestamp": datetime.utcnow().isoformat()
			},
			client_id=client_id
		)
		
		await self.broadcast_to_channel(
			f"{entity_type}_{entity_id}", 
			edit_message, 
			exclude_client=client_id
		)
	
	async def _generate_dashboard_data(
		self, 
		dashboard_type: str, 
		tenant_id: str, 
		user_id: str
	) -> Dict[str, Any]:
		"""Generate real-time dashboard data"""
		# This would integrate with the service layer to get real data
		# For now, return mock data structure
		
		base_data = {
			"tenant_id": tenant_id,
			"generated_at": datetime.utcnow().isoformat(),
			"dashboard_type": dashboard_type
		}
		
		if dashboard_type == "overview":
			return {
				**base_data,
				"active_employees": 150,
				"clocked_in_now": 89,
				"remote_workers": 45,
				"ai_agents_active": 12,
				"fraud_alerts_today": 3,
				"productivity_average": 0.85,
				"recent_activities": []
			}
		elif dashboard_type == "remote_work":
			return {
				**base_data,
				"total_remote_workers": 45,
				"active_sessions": 32,
				"average_productivity": 0.87,
				"burnout_risk_count": 2,
				"top_performers": [],
				"productivity_trends": []
			}
		elif dashboard_type == "ai_agents":
			return {
				**base_data,
				"total_agents": 12,
				"active_agents": 12,
				"tasks_completed_today": 1250,
				"total_cost_today": 125.50,
				"average_performance": 0.94,
				"cost_efficiency": 0.91,
				"agent_types": {
					"conversational_ai": 5,
					"automation_bot": 4,
					"analytics_agent": 2,
					"code_assistant": 1
				}
			}
		
		return base_data
	
	async def _heartbeat_task(self):
		"""Background task for client heartbeat monitoring"""
		while self._running:
			try:
				current_time = datetime.utcnow()
				timeout_threshold = current_time.timestamp() - (self.heartbeat_interval * 2)
				
				# Find stale clients
				stale_clients = []
				for client_id, client in self.clients.items():
					if client.last_ping.timestamp() < timeout_threshold:
						stale_clients.append(client_id)
				
				# Disconnect stale clients
				for client_id in stale_clients:
					logger.info(f"Disconnecting stale client: {client_id}")
					await self.disconnect(client_id)
				
				# Send heartbeat to remaining clients
				heartbeat_msg = WebSocketMessage(
					type="heartbeat",
					channel="system",
					data={"server_time": current_time.isoformat()}
				)
				
				for client_id in list(self.clients.keys()):
					await self._send_to_client(client_id, heartbeat_msg)
			
			except Exception as e:
				logger.error(f"Error in heartbeat task: {str(e)}")
			
			await asyncio.sleep(self.heartbeat_interval)
	
	async def _cleanup_task(self):
		"""Background task for cleanup operations"""
		while self._running:
			try:
				# Clean up empty channel subscriptions
				empty_channels = []
				for channel, subscribers in self.channel_subscribers.items():
					if not subscribers:
						empty_channels.append(channel)
				
				for channel in empty_channels:
					del self.channel_subscribers[channel]
				
				if empty_channels:
					logger.debug(f"Cleaned up {len(empty_channels)} empty channels")
			
			except Exception as e:
				logger.error(f"Error in cleanup task: {str(e)}")
			
			await asyncio.sleep(self.cleanup_interval)
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get WebSocket manager statistics"""
		return {
			"total_clients": len(self.clients),
			"clients_by_tenant": {
				tenant: len(clients) 
				for tenant, clients in self.tenant_clients.items()
			},
			"active_channels": len(self.channel_subscribers),
			"channel_subscribers": {
				channel: len(subscribers)
				for channel, subscribers in self.channel_subscribers.items()
			},
			"running": self._running
		}


# Global WebSocket manager instance
websocket_manager = WebSocketManager()

# Export public interface
__all__ = [
	"WebSocketManager",
	"WebSocketMessage", 
	"RealTimeEvent",
	"websocket_manager"
]
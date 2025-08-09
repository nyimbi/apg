"""
APG Workflow Collaboration Manager

Real-time collaboration system for workflow design.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from uuid import uuid4
import json
import weakref

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

class CollaborationUser(BaseModel):
	"""Represents a collaborating user."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str = Field(..., description="User ID")
	session_id: str = Field(..., description="Session ID")
	username: str = Field(..., description="Username")
	display_name: str = Field(..., description="Display name")
	avatar_url: Optional[str] = Field(default=None, description="Avatar URL")
	color: str = Field(..., description="User color for visual identification")
	
	# State
	cursor_position: Optional[Dict[str, float]] = Field(default=None, description="Current cursor position")
	selected_nodes: List[str] = Field(default_factory=list, description="Currently selected nodes")
	active: bool = Field(default=True, description="Whether user is active")
	
	# Metadata
	joined_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CollaborationChange(BaseModel):
	"""Represents a collaboration change event."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=lambda: str(uuid4()), description="Change ID")
	type: str = Field(..., description="Change type")
	user_id: str = Field(..., description="User who made the change")
	session_id: str = Field(..., description="Session ID")
	data: Dict[str, Any] = Field(..., description="Change data")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CollaborationSession(BaseModel):
	"""Represents a collaboration session."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	session_id: str = Field(..., description="Session ID")
	workflow_id: Optional[str] = Field(default=None, description="Associated workflow ID")
	users: Dict[str, CollaborationUser] = Field(default_factory=dict, description="Active users")
	
	# Settings
	max_users: int = Field(default=10, description="Maximum number of users")
	allow_editing: bool = Field(default=True, description="Whether editing is allowed")
	
	# State
	locked_nodes: Dict[str, str] = Field(default_factory=dict, description="Locked nodes (node_id -> user_id)")
	change_history: List[CollaborationChange] = Field(default_factory=list, description="Recent changes")
	
	# Metadata
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CollaborationManager:
	"""
	Real-time collaboration manager for workflow design.
	
	Features:
	- Multi-user real-time editing
	- User presence and awareness
	- Conflict resolution
	- Change broadcasting
	- Node locking and permissions
	- Chat and comments
	"""
	
	def __init__(self, config):
		self.config = config
		self.sessions: Dict[str, CollaborationSession] = {}
		self.user_colors = [
			"#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
			"#1abc9c", "#e67e22", "#34495e", "#16a085", "#27ae60"
		]
		self.color_index = 0
		self.is_initialized = False
		
		# WebSocket connection management
		self.websocket_connections: Dict[str, Dict[str, Any]] = {}  # session_id -> {user_id -> websocket}
		self.user_session_map: Dict[str, str] = {}  # user_id -> session_id
		self.connection_cleanup_task: Optional[asyncio.Task] = None
		
		logger.info("Collaboration manager initialized")
	
	async def initialize(self) -> None:
		"""Initialize the collaboration manager."""
		try:
			self.is_initialized = True
			
			# Start connection cleanup task
			self.connection_cleanup_task = asyncio.create_task(self._cleanup_dead_connections())
			
			logger.info("Collaboration manager initialization completed")
		except Exception as e:
			logger.error(f"Failed to initialize collaboration manager: {e}")
			raise
	
	async def shutdown(self) -> None:
		"""Shutdown the collaboration manager."""
		try:
			# Cancel cleanup task
			if self.connection_cleanup_task:
				self.connection_cleanup_task.cancel()
				try:
					await self.connection_cleanup_task
				except asyncio.CancelledError:
					pass
			
			# Close all WebSocket connections
			for session_id in list(self.websocket_connections.keys()):
				await self._close_session_connections(session_id)
			
			# Close all sessions
			for session_id in list(self.sessions.keys()):
				await self.close_session(session_id)
			
			self.sessions.clear()
			self.websocket_connections.clear()
			self.user_session_map.clear()
			self.is_initialized = False
			logger.info("Collaboration manager shutdown completed")
		except Exception as e:
			logger.error(f"Error during collaboration manager shutdown: {e}")
	
	async def join_session(self, session_id: str, user_id: str, user_info: Optional[Dict[str, Any]] = None) -> CollaborationUser:
		"""Add user to collaboration session."""
		try:
			if session_id not in self.sessions:
				self.sessions[session_id] = CollaborationSession(session_id=session_id)
			
			session = self.sessions[session_id]
			
			# Check user limit
			if len(session.users) >= session.max_users:
				raise ValueError(f"Session {session_id} has reached maximum user limit")
			
			# Create user object
			user_color = self._get_next_user_color()
			user = CollaborationUser(
				user_id=user_id,
				session_id=session_id,
				username=user_info.get('username', f'User{user_id[:8]}') if user_info else f'User{user_id[:8]}',
				display_name=user_info.get('display_name', f'User {user_id[:8]}') if user_info else f'User {user_id[:8]}',
				avatar_url=user_info.get('avatar_url') if user_info else None,
				color=user_color
			)
			
			# Add to session
			session.users[user_id] = user
			session.last_activity = datetime.now(timezone.utc)
			
			# Initialize WebSocket connections for session if needed
			if session_id not in self.websocket_connections:
				self.websocket_connections[session_id] = {}
			
			# Track user session mapping
			self.user_session_map[user_id] = session_id
			
			# Broadcast user join
			await self._broadcast_to_session(session_id, {
				'type': 'user_joined',
				'user': user.model_dump(),
				'timestamp': datetime.now(timezone.utc).isoformat()
			}, exclude_user=user_id)
			
			logger.info(f"User {user_id} joined collaboration session {session_id}")
			return user
			
		except Exception as e:
			logger.error(f"Failed to join session: {e}")
			raise
	
	async def leave_session(self, session_id: str, user_id: str) -> None:
		"""Remove user from collaboration session."""
		try:
			if session_id not in self.sessions:
				return
			
			session = self.sessions[session_id]
			
			if user_id in session.users:
				user = session.users[user_id]
				
				# Release any locked nodes
				nodes_to_unlock = [node_id for node_id, lock_user in session.locked_nodes.items() if lock_user == user_id]
				for node_id in nodes_to_unlock:
					del session.locked_nodes[node_id]
				
				# Close user's WebSocket connection
				await self._close_user_connection(session_id, user_id)
				
				# Remove user from session mapping
				if user_id in self.user_session_map:
					del self.user_session_map[user_id]
				
				# Remove user
				del session.users[user_id]
				
				# Broadcast user leave
				await self._broadcast_to_session(session_id, {
					'type': 'user_left',
					'user_id': user_id,
					'username': user.username,
					'unlocked_nodes': nodes_to_unlock,
					'timestamp': datetime.now(timezone.utc).isoformat()
				}, exclude_user=user_id)
				
				# Close session if empty
				if not session.users:
					await self.close_session(session_id)
				
				logger.info(f"User {user_id} left collaboration session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to leave session: {e}")
	
	async def close_session(self, session_id: str) -> None:
		"""Close a collaboration session."""
		try:
			if session_id in self.sessions:
				session = self.sessions[session_id]
				
				# Notify all users
				await self._broadcast_to_session(session_id, {
					'type': 'session_closed',
					'timestamp': datetime.now(timezone.utc).isoformat()
				})
				
				# Remove session
				del self.sessions[session_id]
				
				logger.info(f"Closed collaboration session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to close session: {e}")
	
	async def broadcast_change(self, session_id: str, change_data: Dict[str, Any]) -> None:
		"""Broadcast a change to all users in session."""
		try:
			if session_id not in self.sessions:
				return
			
			session = self.sessions[session_id]
			
			# Create change object
			change = CollaborationChange(
				type=change_data.get('type', 'unknown'),
				user_id=change_data.get('user_id', 'system'),
				session_id=session_id,
				data=change_data
			)
			
			# Add to history
			session.change_history.append(change)
			
			# Limit history size
			if len(session.change_history) > 100:
				session.change_history = session.change_history[-100:]
			
			# Update session activity
			session.last_activity = datetime.now(timezone.utc)
			
			# Broadcast to all users except the originator
			await self._broadcast_to_session(session_id, change.model_dump(), exclude_user=change.user_id)
			
		except Exception as e:
			logger.error(f"Failed to broadcast change: {e}")
	
	async def update_user_presence(self, session_id: str, user_id: str, presence_data: Dict[str, Any]) -> None:
		"""Update user presence information."""
		try:
			if session_id not in self.sessions:
				return
			
			session = self.sessions[session_id]
			
			if user_id not in session.users:
				return
			
			user = session.users[user_id]
			
			# Update presence data
			if 'cursor_position' in presence_data:
				user.cursor_position = presence_data['cursor_position']
			
			if 'selected_nodes' in presence_data:
				user.selected_nodes = presence_data['selected_nodes']
			
			user.last_activity = datetime.now(timezone.utc)
			
			# Broadcast presence update
			await self._broadcast_to_session(session_id, {
				'type': 'presence_update',
				'user_id': user_id,
				'presence': {
					'cursor_position': user.cursor_position,
					'selected_nodes': user.selected_nodes
				},
				'timestamp': datetime.now(timezone.utc).isoformat()
			}, exclude_user=user_id)
			
		except Exception as e:
			logger.error(f"Failed to update user presence: {e}")
	
	async def lock_node(self, session_id: str, user_id: str, node_id: str) -> bool:
		"""Lock a node for exclusive editing."""
		try:
			if session_id not in self.sessions:
				return False
			
			session = self.sessions[session_id]
			
			# Check if node is already locked
			if node_id in session.locked_nodes:
				return session.locked_nodes[node_id] == user_id
			
			# Lock the node
			session.locked_nodes[node_id] = user_id
			
			# Broadcast lock
			await self._broadcast_to_session(session_id, {
				'type': 'node_locked',
				'node_id': node_id,
				'user_id': user_id,
				'timestamp': datetime.now(timezone.utc).isoformat()
			}, exclude_user=user_id)
			
			return True
			
		except Exception as e:
			logger.error(f"Failed to lock node: {e}")
			return False
	
	async def unlock_node(self, session_id: str, user_id: str, node_id: str) -> bool:
		"""Unlock a node."""
		try:
			if session_id not in self.sessions:
				return False
			
			session = self.sessions[session_id]
			
			# Check if user owns the lock
			if node_id not in session.locked_nodes or session.locked_nodes[node_id] != user_id:
				return False
			
			# Unlock the node
			del session.locked_nodes[node_id]
			
			# Broadcast unlock
			await self._broadcast_to_session(session_id, {
				'type': 'node_unlocked',
				'node_id': node_id,
				'user_id': user_id,
				'timestamp': datetime.now(timezone.utc).isoformat()
			}, exclude_user=user_id)
			
			return True
			
		except Exception as e:
			logger.error(f"Failed to unlock node: {e}")
			return False
	
	async def get_session_collaborators(self, session_id: str) -> List[Dict[str, Any]]:
		"""Get list of collaborators in session."""
		try:
			if session_id not in self.sessions:
				return []
			
			session = self.sessions[session_id]
			
			return [
				{
					'user_id': user.user_id,
					'username': user.username,
					'display_name': user.display_name,
					'avatar_url': user.avatar_url,
					'color': user.color,
					'active': user.active,
					'joined_at': user.joined_at.isoformat(),
					'last_activity': user.last_activity.isoformat()
				}
				for user in session.users.values()
			]
			
		except Exception as e:
			logger.error(f"Failed to get session collaborators: {e}")
			return []
	
	async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
		"""Get collaboration session information."""
		try:
			if session_id not in self.sessions:
				return None
			
			session = self.sessions[session_id]
			
			return {
				'session_id': session.session_id,
				'workflow_id': session.workflow_id,
				'user_count': len(session.users),
				'max_users': session.max_users,
				'allow_editing': session.allow_editing,
				'locked_nodes': list(session.locked_nodes.keys()),
				'created_at': session.created_at.isoformat(),
				'last_activity': session.last_activity.isoformat(),
				'users': await self.get_session_collaborators(session_id)
			}
			
		except Exception as e:
			logger.error(f"Failed to get session info: {e}")
			return None
	
	async def add_comment(self, session_id: str, user_id: str, node_id: str, comment: str) -> Dict[str, Any]:
		"""Add a comment to a node."""
		try:
			comment_data = {
				'id': str(uuid4()),
				'user_id': user_id,
				'node_id': node_id,
				'comment': comment,
				'timestamp': datetime.now(timezone.utc).isoformat()
			}
			
			# Broadcast comment
			await self._broadcast_to_session(session_id, {
				'type': 'comment_added',
				'comment': comment_data
			}, exclude_user=user_id)
			
			return comment_data
			
		except Exception as e:
			logger.error(f"Failed to add comment: {e}")
			raise
	
	async def get_change_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
		"""Get recent change history for session."""
		try:
			if session_id not in self.sessions:
				return []
			
			session = self.sessions[session_id]
			
			# Get recent changes
			recent_changes = session.change_history[-limit:] if len(session.change_history) > limit else session.change_history
			
			return [change.model_dump() for change in recent_changes]
			
		except Exception as e:
			logger.error(f"Failed to get change history: {e}")
			return []
	
	# Private methods
	
	def _get_next_user_color(self) -> str:
		"""Get next color for user identification."""
		color = self.user_colors[self.color_index % len(self.user_colors)]
		self.color_index += 1
		return color
	
	async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None) -> None:
		"""Broadcast message to all users in session via WebSocket connections."""
		try:
			if session_id not in self.sessions:
				return
			
			session = self.sessions[session_id]
			session_connections = self.websocket_connections.get(session_id, {})
			
			# Get recipients (all users except excluded)
			recipients = [
				user_id for user_id in session.users.keys()
				if user_id != exclude_user
			]
			
			if not recipients:
				return
			
			# Prepare message with metadata
			broadcast_message = {
				'session_id': session_id,
				'timestamp': datetime.now(timezone.utc).isoformat(),
				**message
			}
			
			message_json = json.dumps(broadcast_message)
			sent_count = 0
			failed_connections = []
			
			# Send to all recipient WebSocket connections
			for user_id in recipients:
				websocket = session_connections.get(user_id)
				if websocket:
					try:
						# Check if connection is still alive
						if hasattr(websocket, 'closed') and websocket.closed:
							failed_connections.append(user_id)
							continue
						
						# Send message
						await websocket.send(message_json)
						sent_count += 1
						
					except Exception as e:
						logger.warning(f"Failed to send message to user {user_id}: {e}")
						failed_connections.append(user_id)
			
			# Clean up failed connections
			for user_id in failed_connections:
				await self._close_user_connection(session_id, user_id)
			
			if sent_count > 0:
				logger.debug(f"Broadcasted {message.get('type', 'unknown')} to {sent_count}/{len(recipients)} users in session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to broadcast to session: {e}")
	
	async def cleanup_inactive_sessions(self, max_idle_time: int = 3600) -> None:
		"""Clean up inactive sessions."""
		try:
			current_time = datetime.now(timezone.utc)
			sessions_to_remove = []
			
			for session_id, session in self.sessions.items():
				idle_time = (current_time - session.last_activity).total_seconds()
				if idle_time > max_idle_time:
					sessions_to_remove.append(session_id)
			
			for session_id in sessions_to_remove:
				await self.close_session(session_id)
				logger.info(f"Cleaned up inactive session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to cleanup inactive sessions: {e}")
	
	async def cleanup_inactive_users(self, max_idle_time: int = 300) -> None:
		"""Clean up inactive users from sessions."""
		try:
			current_time = datetime.now(timezone.utc)
			
			for session_id, session in self.sessions.items():
				users_to_remove = []
				
				for user_id, user in session.users.items():
					idle_time = (current_time - user.last_activity).total_seconds()
					if idle_time > max_idle_time:
						users_to_remove.append(user_id)
				
				for user_id in users_to_remove:
					await self.leave_session(session_id, user_id)
					logger.info(f"Cleaned up inactive user {user_id} from session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to cleanup inactive users: {e}")
	
	# WebSocket connection management methods
	
	async def register_websocket(self, session_id: str, user_id: str, websocket: Any) -> None:
		"""Register a WebSocket connection for a user in a session."""
		try:
			if session_id not in self.websocket_connections:
				self.websocket_connections[session_id] = {}
			
			# Close existing connection if any
			await self._close_user_connection(session_id, user_id)
			
			# Register new connection
			self.websocket_connections[session_id][user_id] = websocket
			
			# Send initial session state to the new connection
			await self._send_session_state(session_id, user_id)
			
			logger.info(f"Registered WebSocket for user {user_id} in session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to register WebSocket: {e}")
			raise
	
	async def unregister_websocket(self, session_id: str, user_id: str) -> None:
		"""Unregister a WebSocket connection for a user."""
		try:
			await self._close_user_connection(session_id, user_id)
			logger.info(f"Unregistered WebSocket for user {user_id} in session {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to unregister WebSocket: {e}")
	
	async def _close_user_connection(self, session_id: str, user_id: str) -> None:
		"""Close a specific user's WebSocket connection."""
		try:
			session_connections = self.websocket_connections.get(session_id, {})
			websocket = session_connections.get(user_id)
			
			if websocket:
				try:
					if hasattr(websocket, 'close') and not getattr(websocket, 'closed', True):
						await websocket.close()
				except Exception as e:
					logger.debug(f"Error closing WebSocket for user {user_id}: {e}")
				
				# Remove from connections
				del session_connections[user_id]
				
				# Clean up empty session connections
				if not session_connections and session_id in self.websocket_connections:
					del self.websocket_connections[session_id]
		
		except Exception as e:
			logger.error(f"Failed to close user connection: {e}")
	
	async def _close_session_connections(self, session_id: str) -> None:
		"""Close all WebSocket connections for a session."""
		try:
			session_connections = self.websocket_connections.get(session_id, {})
			
			for user_id in list(session_connections.keys()):
				await self._close_user_connection(session_id, user_id)
			
			if session_id in self.websocket_connections:
				del self.websocket_connections[session_id]
				
		except Exception as e:
			logger.error(f"Failed to close session connections: {e}")
	
	async def _send_session_state(self, session_id: str, user_id: str) -> None:
		"""Send current session state to a specific user."""
		try:
			session_info = await self.get_session_info(session_id)
			if session_info:
				session_connections = self.websocket_connections.get(session_id, {})
				websocket = session_connections.get(user_id)
				
				if websocket:
					state_message = {
						'type': 'session_state',
						'session_info': session_info,
						'timestamp': datetime.now(timezone.utc).isoformat()
					}
					
					await websocket.send(json.dumps(state_message))
					
		except Exception as e:
			logger.error(f"Failed to send session state: {e}")
	
	async def _cleanup_dead_connections(self) -> None:
		"""Background task to clean up dead WebSocket connections."""
		while self.is_initialized:
			try:
				await asyncio.sleep(30)  # Check every 30 seconds
				
				dead_connections = []
				
				for session_id, session_connections in self.websocket_connections.items():
					for user_id, websocket in session_connections.items():
						try:
							# Check if connection is dead
							if hasattr(websocket, 'closed') and websocket.closed:
								dead_connections.append((session_id, user_id))
							elif hasattr(websocket, 'ping'):
								# Send ping to check connection health
								await websocket.ping()
						except Exception:
							dead_connections.append((session_id, user_id))
				
				# Clean up dead connections
				for session_id, user_id in dead_connections:
					await self._close_user_connection(session_id, user_id)
					logger.debug(f"Cleaned up dead connection for user {user_id} in session {session_id}")
			
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Error in connection cleanup task: {e}")
	
	async def send_direct_message(self, session_id: str, user_id: str, message: Dict[str, Any]) -> bool:
		"""Send a direct message to a specific user."""
		try:
			session_connections = self.websocket_connections.get(session_id, {})
			websocket = session_connections.get(user_id)
			
			if not websocket:
				return False
			
			direct_message = {
				'type': 'direct_message',
				'session_id': session_id,
				'timestamp': datetime.now(timezone.utc).isoformat(),
				**message
			}
			
			await websocket.send(json.dumps(direct_message))
			return True
			
		except Exception as e:
			logger.error(f"Failed to send direct message: {e}")
			return False
	
	def get_active_connections_count(self, session_id: str) -> int:
		"""Get the number of active WebSocket connections for a session."""
		return len(self.websocket_connections.get(session_id, {}))
	
	def get_user_connection_status(self, session_id: str, user_id: str) -> Dict[str, Any]:
		"""Get connection status for a specific user."""
		session_connections = self.websocket_connections.get(session_id, {})
		websocket = session_connections.get(user_id)
		
		return {
			'connected': websocket is not None,
			'connection_alive': websocket is not None and not getattr(websocket, 'closed', True) if websocket else False,
			'session_id': session_id,
			'user_id': user_id
		}
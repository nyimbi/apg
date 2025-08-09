"""
Advanced Real-Time Collaboration Module

Provides enhanced real-time collaboration features:
- Multi-user 3D workspace sharing
- Real-time cursor and avatar tracking
- Collaborative editing with conflict resolution
- Voice/video conferencing integration
- Screen sharing and presentation modes
- Advanced presence awareness
- Collaborative AI assistance

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .visualization_3d import Vector3D, Node3D, Edge3D
from .database import get_async_db_session


class PresenceStatus(str, Enum):
	"""User presence status"""
	ONLINE = "online"
	AWAY = "away"
	BUSY = "busy"
	INVISIBLE = "invisible"
	OFFLINE = "offline"


class InteractionType(str, Enum):
	"""Types of user interactions"""
	CURSOR_MOVE = "cursor_move"
	NODE_SELECT = "node_select"
	NODE_DRAG = "node_drag"
	NODE_EDIT = "node_edit"
	EDGE_CREATE = "edge_create"
	EDGE_DELETE = "edge_delete"
	VIEWPORT_CHANGE = "viewport_change"
	TOOL_SELECT = "tool_select"
	CHAT_MESSAGE = "chat_message"
	VOICE_SPEAK = "voice_speak"
	ANNOTATION = "annotation"


class ConflictResolutionStrategy(str, Enum):
	"""Conflict resolution strategies"""
	LAST_WRITE_WINS = "last_write_wins"
	FIRST_WRITE_WINS = "first_write_wins"
	MERGE_CHANGES = "merge_changes"
	USER_RESOLVE = "user_resolve"
	AI_RESOLVE = "ai_resolve"


class CollaborationMode(str, Enum):
	"""Collaboration modes"""
	REAL_TIME = "real_time"
	TURN_BASED = "turn_based"
	OBSERVER = "observer"
	PRESENTATION = "presentation"
	REVIEW = "review"


@dataclass
class UserAvatar:
	"""3D user avatar representation"""
	user_id: str
	name: str
	position: Vector3D = field(default_factory=Vector3D)
	rotation: Vector3D = field(default_factory=Vector3D)
	color: str = "#2196F3"
	avatar_type: str = "cursor"  # cursor, character, robot
	scale: float = 1.0
	visible: bool = True
	animation: str = "idle"
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserCursor:
	"""User cursor/pointer in 3D space"""
	user_id: str
	position: Vector3D = field(default_factory=Vector3D)
	direction: Vector3D = field(default_factory=Vector3D)
	target_object: Optional[str] = None
	interaction_type: Optional[InteractionType] = None
	visible: bool = True
	color: str = "#FF5722"
	label: str = ""
	timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserPresence:
	"""User presence information"""
	user_id: str
	session_id: str
	name: str
	email: str
	status: PresenceStatus = PresenceStatus.ONLINE
	avatar: UserAvatar = field(default_factory=lambda: UserAvatar("", ""))
	cursor: UserCursor = field(default_factory=lambda: UserCursor(""))
	current_tool: str = "select"
	viewport: Dict[str, Any] = field(default_factory=dict)
	permissions: List[str] = field(default_factory=list)
	joined_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	device_info: Dict[str, Any] = field(default_factory=dict)


class ChangeOperation(BaseModel):
	"""Represents a workflow change operation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	user_id: str
	session_id: str
	operation_type: str  # create, update, delete, move
	target_type: str  # node, edge, property
	target_id: str
	changes: Dict[str, Any] = Field(default_factory=dict)
	previous_state: Optional[Dict[str, Any]] = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	applied: bool = False
	conflicts: List[str] = Field(default_factory=list)


class CollaborationSession:
	"""Manages a collaborative workflow session"""
	
	def __init__(self, workflow_id: str, session_id: str):
		self.workflow_id = workflow_id
		self.session_id = session_id
		self.users: Dict[str, UserPresence] = {}
		self.active_operations: List[ChangeOperation] = []
		self.operation_history: List[ChangeOperation] = []
		self.locks: Dict[str, str] = {}  # object_id -> user_id
		self.conflict_resolution = ConflictResolutionStrategy.LAST_WRITE_WINS
		self.mode = CollaborationMode.REAL_TIME
		self.created_at = datetime.utcnow()
		self.last_activity = datetime.utcnow()
		
		# Communication channels
		self.chat_messages: List[Dict[str, Any]] = []
		self.voice_channels: Dict[str, Any] = {}
		self.screen_sharing: Optional[Dict[str, Any]] = None
		
		# AI collaboration assistant
		self.ai_assistant_enabled = True
		self.ai_suggestions: List[Dict[str, Any]] = []
		
		# Performance tracking
		self.operation_count = 0
		self.conflict_count = 0
		self.sync_latency_ms = 0
	
	async def add_user(self, user_presence: UserPresence) -> bool:
		"""Add user to collaboration session"""
		try:
			self.users[user_presence.user_id] = user_presence
			self.last_activity = datetime.utcnow()
			
			# Initialize avatar and cursor
			user_presence.avatar.user_id = user_presence.user_id
			user_presence.cursor.user_id = user_presence.user_id
			
			# Assign unique color
			user_presence.avatar.color = self._generate_user_color(user_presence.user_id)
			user_presence.cursor.color = user_presence.avatar.color
			
			# Store in database
			await self._store_user_presence(user_presence)
			
			# Broadcast user joined event
			await self._broadcast_event("user_joined", {
				"user_id": user_presence.user_id,
				"name": user_presence.name,
				"avatar": user_presence.avatar.__dict__,
				"timestamp": datetime.utcnow().isoformat()
			})
			
			return True
			
		except Exception as e:
			print(f"Add user error: {e}")
			return False
	
	async def remove_user(self, user_id: str) -> bool:
		"""Remove user from collaboration session"""
		try:
			if user_id in self.users:
				user = self.users[user_id]
				
				# Release any locks held by this user
				await self._release_user_locks(user_id)
				
				# Remove user
				del self.users[user_id]
				self.last_activity = datetime.utcnow()
				
				# Broadcast user left event
				await self._broadcast_event("user_left", {
					"user_id": user_id,
					"name": user.name,
					"timestamp": datetime.utcnow().isoformat()
				})
				
				return True
			
			return False
			
		except Exception as e:
			print(f"Remove user error: {e}")
			return False
	
	async def update_user_presence(self, user_id: str, updates: Dict[str, Any]) -> bool:
		"""Update user presence information"""
		try:
			if user_id not in self.users:
				return False
			
			user = self.users[user_id]
			user.last_activity = datetime.utcnow()
			
			# Update avatar
			if "avatar" in updates:
				avatar_updates = updates["avatar"]
				if "position" in avatar_updates:
					user.avatar.position = Vector3D(**avatar_updates["position"])
				if "rotation" in avatar_updates:
					user.avatar.rotation = Vector3D(**avatar_updates["rotation"])
				if "animation" in avatar_updates:
					user.avatar.animation = avatar_updates["animation"]
			
			# Update cursor
			if "cursor" in updates:
				cursor_updates = updates["cursor"]
				if "position" in cursor_updates:
					user.cursor.position = Vector3D(**cursor_updates["position"])
				if "direction" in cursor_updates:
					user.cursor.direction = Vector3D(**cursor_updates["direction"])
				if "target_object" in cursor_updates:
					user.cursor.target_object = cursor_updates["target_object"]
				if "interaction_type" in cursor_updates:
					user.cursor.interaction_type = InteractionType(cursor_updates["interaction_type"])
			
			# Update status
			if "status" in updates:
				user.status = PresenceStatus(updates["status"])
			
			# Update viewport
			if "viewport" in updates:
				user.viewport = updates["viewport"]
			
			# Update current tool
			if "current_tool" in updates:
				user.current_tool = updates["current_tool"]
			
			# Broadcast presence update
			await self._broadcast_event("presence_updated", {
				"user_id": user_id,
				"updates": updates,
				"timestamp": datetime.utcnow().isoformat()
			})
			
			return True
			
		except Exception as e:
			print(f"Update presence error: {e}")
			return False
	
	async def apply_operation(self, operation: ChangeOperation) -> Tuple[bool, List[str]]:
		"""Apply change operation with conflict detection"""
		try:
			conflicts = []
			
			# Check for conflicts
			conflicts.extend(await self._detect_conflicts(operation))
			
			if conflicts and self.conflict_resolution == ConflictResolutionStrategy.USER_RESOLVE:
				# Add to pending operations for user resolution
				operation.conflicts = conflicts
				self.active_operations.append(operation)
				return False, conflicts
			
			# Apply conflict resolution
			if conflicts:
				resolved_operation = await self._resolve_conflicts(operation, conflicts)
				if resolved_operation:
					operation = resolved_operation
				else:
					return False, conflicts
			
			# Apply operation
			success = await self._execute_operation(operation)
			
			if success:
				operation.applied = True
				operation.timestamp = datetime.utcnow()
				
				# Add to history
				self.operation_history.append(operation)
				self.operation_count += 1
				
				# Remove from active operations if it was there
				self.active_operations = [op for op in self.active_operations if op.id != operation.id]
				
				# Broadcast operation to other users
				await self._broadcast_operation(operation)
				
				# Check for AI suggestions
				if self.ai_assistant_enabled:
					await self._generate_ai_suggestions(operation)
			
			return success, conflicts
			
		except Exception as e:
			print(f"Apply operation error: {e}")
			return False, [str(e)]
	
	async def _detect_conflicts(self, operation: ChangeOperation) -> List[str]:
		"""Detect conflicts with other operations"""
		conflicts = []
		
		try:
			# Check for concurrent modifications to the same object
			recent_threshold = datetime.utcnow() - timedelta(seconds=30)
			
			for existing_op in self.operation_history:
				if (existing_op.target_id == operation.target_id and
					existing_op.user_id != operation.user_id and
					existing_op.timestamp > recent_threshold):
					
					conflicts.append(f"Concurrent modification by {existing_op.user_id}")
			
			# Check for locks
			if operation.target_id in self.locks:
				lock_owner = self.locks[operation.target_id]
				if lock_owner != operation.user_id:
					conflicts.append(f"Object locked by {lock_owner}")
			
			# Check for dependency conflicts
			if operation.operation_type == "delete":
				dependencies = await self._get_object_dependencies(operation.target_id)
				if dependencies:
					conflicts.append(f"Object has dependencies: {dependencies}")
			
			return conflicts
			
		except Exception as e:
			print(f"Conflict detection error: {e}")
			return [str(e)]
	
	async def _resolve_conflicts(self, operation: ChangeOperation, conflicts: List[str]) -> Optional[ChangeOperation]:
		"""Resolve conflicts based on strategy"""
		try:
			if self.conflict_resolution == ConflictResolutionStrategy.LAST_WRITE_WINS:
				# Simply apply the latest operation
				return operation
			
			elif self.conflict_resolution == ConflictResolutionStrategy.FIRST_WRITE_WINS:
				# Reject the operation if there are conflicts
				return None
			
			elif self.conflict_resolution == ConflictResolutionStrategy.MERGE_CHANGES:
				# Attempt to merge changes
				return await self._merge_operation_changes(operation)
			
			elif self.conflict_resolution == ConflictResolutionStrategy.AI_RESOLVE:
				# Use AI to resolve conflicts
				return await self._ai_resolve_conflicts(operation, conflicts)
			
			return operation
			
		except Exception as e:
			print(f"Conflict resolution error: {e}")
			return None
	
	async def _execute_operation(self, operation: ChangeOperation) -> bool:
		"""Execute the change operation"""
		try:
			# This would interface with the actual workflow data
			# For now, we'll simulate the operation
			
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				if operation.operation_type == "create":
					# Create new object
					if operation.target_type == "node":
						await session.execute(
							text("""
							INSERT INTO workflow_collaboration_ops (
								id, workflow_id, session_id, user_id, operation_type,
								target_type, target_id, changes, timestamp, applied
							) VALUES (
								:id, :workflow_id, :session_id, :user_id, :operation_type,
								:target_type, :target_id, :changes, :timestamp, :applied
							)
							"""),
							{
								"id": operation.id,
								"workflow_id": self.workflow_id,
								"session_id": self.session_id,
								"user_id": operation.user_id,
								"operation_type": operation.operation_type,
								"target_type": operation.target_type,
								"target_id": operation.target_id,
								"changes": json.dumps(operation.changes),
								"timestamp": operation.timestamp,
								"applied": True
							}
						)
				
				elif operation.operation_type == "update":
					# Update existing object
					await session.execute(
						text("""
						INSERT INTO workflow_collaboration_ops (
							id, workflow_id, session_id, user_id, operation_type,
							target_type, target_id, changes, timestamp, applied
						) VALUES (
							:id, :workflow_id, :session_id, :user_id, :operation_type,
							:target_type, :target_id, :changes, :timestamp, :applied
						)
						"""),
						{
							"id": operation.id,
							"workflow_id": self.workflow_id,
							"session_id": self.session_id,
							"user_id": operation.user_id,
							"operation_type": operation.operation_type,
							"target_type": operation.target_type,
							"target_id": operation.target_id,
							"changes": json.dumps(operation.changes),
							"timestamp": operation.timestamp,
							"applied": True
						}
					)
				
				elif operation.operation_type == "delete":
					# Delete object
					await session.execute(
						text("""
						INSERT INTO workflow_collaboration_ops (
							id, workflow_id, session_id, user_id, operation_type,
							target_type, target_id, changes, timestamp, applied
						) VALUES (
							:id, :workflow_id, :session_id, :user_id, :operation_type,
							:target_type, :target_id, :changes, :timestamp, :applied
						)
						"""),
						{
							"id": operation.id,
							"workflow_id": self.workflow_id,
							"session_id": self.session_id,
							"user_id": operation.user_id,
							"operation_type": operation.operation_type,
							"target_type": operation.target_type,
							"target_id": operation.target_id,
							"changes": json.dumps(operation.changes),
							"timestamp": operation.timestamp,
							"applied": True
						}
					)
				
				await session.commit()
			
			return True
			
		except Exception as e:
			print(f"Execute operation error: {e}")
			return False
	
	async def _merge_operation_changes(self, operation: ChangeOperation) -> Optional[ChangeOperation]:
		"""Merge changes from conflicting operations"""
		try:
			# Find the most recent operation on the same target
			recent_ops = [
				op for op in self.operation_history
				if (op.target_id == operation.target_id and
					op.timestamp > datetime.utcnow() - timedelta(seconds=30))
			]
			
			if not recent_ops:
				return operation
			
			latest_op = max(recent_ops, key=lambda x: x.timestamp)
			
			# Merge changes (simple field-level merge)
			merged_changes = latest_op.changes.copy()
			merged_changes.update(operation.changes)
			
			# Create merged operation
			merged_operation = ChangeOperation(
				user_id=operation.user_id,
				session_id=operation.session_id,
				operation_type=operation.operation_type,
				target_type=operation.target_type,
				target_id=operation.target_id,
				changes=merged_changes,
				previous_state=latest_op.changes
			)
			
			return merged_operation
			
		except Exception as e:
			print(f"Merge changes error: {e}")
			return None
	
	async def _ai_resolve_conflicts(self, operation: ChangeOperation, conflicts: List[str]) -> Optional[ChangeOperation]:
		"""Use AI to resolve conflicts"""
		try:
			# This would integrate with an AI service
			# For now, we'll implement a simple heuristic
			
			# Prioritize certain types of operations
			priority_operations = ["create", "update"]
			
			if operation.operation_type in priority_operations:
				# Create a modified operation that resolves conflicts
				resolved_changes = operation.changes.copy()
				
				# Add conflict resolution metadata
				resolved_changes["_ai_resolved"] = True
				resolved_changes["_original_conflicts"] = conflicts
				resolved_changes["_resolution_strategy"] = "ai_merge"
				
				resolved_operation = ChangeOperation(
					user_id=operation.user_id,
					session_id=operation.session_id,
					operation_type=operation.operation_type,
					target_type=operation.target_type,
					target_id=operation.target_id,
					changes=resolved_changes,
					previous_state=operation.previous_state
				)
				
				return resolved_operation
			
			return None
			
		except Exception as e:
			print(f"AI conflict resolution error: {e}")
			return None
	
	async def _get_object_dependencies(self, object_id: str) -> List[str]:
		"""Get object dependencies"""
		try:
			# This would check for edges connected to a node, etc.
			# For now, return empty list
			return []
			
		except Exception as e:
			print(f"Get dependencies error: {e}")
			return []
	
	async def _broadcast_event(self, event_type: str, data: Dict[str, Any]):
		"""Broadcast event to all users in session"""
		try:
			event = {
				"type": event_type,
				"session_id": self.session_id,
				"workflow_id": self.workflow_id,
				"data": data,
				"timestamp": datetime.utcnow().isoformat()
			}
			
			# In a real implementation, this would use WebSocket broadcasting
			print(f"Broadcasting event: {event}")
			
		except Exception as e:
			print(f"Broadcast event error: {e}")
	
	async def _broadcast_operation(self, operation: ChangeOperation):
		"""Broadcast operation to other users"""
		try:
			operation_data = {
				"id": operation.id,
				"user_id": operation.user_id,
				"operation_type": operation.operation_type,
				"target_type": operation.target_type,
				"target_id": operation.target_id,
				"changes": operation.changes,
				"timestamp": operation.timestamp.isoformat()
			}
			
			await self._broadcast_event("operation_applied", operation_data)
			
		except Exception as e:
			print(f"Broadcast operation error: {e}")
	
	async def _generate_ai_suggestions(self, operation: ChangeOperation):
		"""Generate AI suggestions based on operation"""
		try:
			# This would use AI to generate suggestions
			suggestions = []
			
			if operation.operation_type == "create" and operation.target_type == "node":
				# Suggest connecting to nearby nodes
				suggestions.append({
					"type": "connection_suggestion",
					"message": "Consider connecting this node to nearby workflow steps",
					"confidence": 0.8,
					"actions": ["auto_connect"]
				})
			
			elif operation.operation_type == "update":
				# Suggest optimizations
				suggestions.append({
					"type": "optimization_suggestion",
					"message": "This change might affect workflow performance",
					"confidence": 0.6,
					"actions": ["analyze_performance"]
				})
			
			self.ai_suggestions.extend(suggestions)
			
			# Broadcast suggestions
			if suggestions:
				await self._broadcast_event("ai_suggestions", {
					"suggestions": suggestions,
					"related_operation": operation.id
				})
			
		except Exception as e:
			print(f"AI suggestions error: {e}")
	
	async def _store_user_presence(self, user_presence: UserPresence):
		"""Store user presence in database"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				await session.execute(
					text("""
					INSERT INTO workflow_user_presence (
						user_id, session_id, workflow_id, name, email, status,
						avatar_data, cursor_data, current_tool, viewport,
						permissions, joined_at, last_activity, device_info
					) VALUES (
						:user_id, :session_id, :workflow_id, :name, :email, :status,
						:avatar_data, :cursor_data, :current_tool, :viewport,
						:permissions, :joined_at, :last_activity, :device_info
					)
					ON CONFLICT (user_id, session_id) 
					DO UPDATE SET 
						status = EXCLUDED.status,
						avatar_data = EXCLUDED.avatar_data,
						cursor_data = EXCLUDED.cursor_data,
						current_tool = EXCLUDED.current_tool,
						viewport = EXCLUDED.viewport,
						last_activity = EXCLUDED.last_activity
					"""),
					{
						"user_id": user_presence.user_id,
						"session_id": user_presence.session_id,
						"workflow_id": self.workflow_id,
						"name": user_presence.name,
						"email": user_presence.email,
						"status": user_presence.status.value,
						"avatar_data": json.dumps(user_presence.avatar.__dict__),
						"cursor_data": json.dumps(user_presence.cursor.__dict__),
						"current_tool": user_presence.current_tool,
						"viewport": json.dumps(user_presence.viewport),
						"permissions": json.dumps(user_presence.permissions),
						"joined_at": user_presence.joined_at,
						"last_activity": user_presence.last_activity,
						"device_info": json.dumps(user_presence.device_info)
					}
				)
				await session.commit()
				
		except Exception as e:
			print(f"Store user presence error: {e}")
	
	async def _release_user_locks(self, user_id: str):
		"""Release all locks held by user"""
		try:
			released_objects = []
			for object_id, lock_owner in list(self.locks.items()):
				if lock_owner == user_id:
					del self.locks[object_id]
					released_objects.append(object_id)
			
			if released_objects:
				await self._broadcast_event("locks_released", {
					"user_id": user_id,
					"objects": released_objects
				})
				
		except Exception as e:
			print(f"Release locks error: {e}")
	
	def _generate_user_color(self, user_id: str) -> str:
		"""Generate unique color for user"""
		# Simple hash-based color generation
		hash_value = hash(user_id)
		colors = [
			"#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0",
			"#607D8B", "#795548", "#009688", "#3F51B5", "#E91E63"
		]
		return colors[abs(hash_value) % len(colors)]


class AdvancedCollaborationManager:
	"""Advanced collaboration management system"""
	
	def __init__(self):
		self.sessions: Dict[str, CollaborationSession] = {}
		self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
		self.voice_channels: Dict[str, Dict[str, Any]] = {}
		self.screen_sharing_sessions: Dict[str, Dict[str, Any]] = {}
		
		# AI collaboration features
		self.ai_mediator_enabled = True
		self.ai_suggestion_threshold = 0.7
		
		# Performance settings
		self.max_concurrent_operations = 50
		self.operation_batch_size = 10
		self.presence_update_rate = 30  # Hz
	
	async def create_collaboration_session(self, workflow_id: str, creator_user_id: str, config: Dict[str, Any]) -> str:
		"""Create new collaboration session"""
		try:
			session_id = uuid7str()
			session = CollaborationSession(workflow_id, session_id)
			
			# Configure session
			session.mode = CollaborationMode(config.get("mode", "real_time"))
			session.conflict_resolution = ConflictResolutionStrategy(config.get("conflict_resolution", "last_write_wins"))
			session.ai_assistant_enabled = config.get("ai_assistant", True)
			
			# Store session
			self.sessions[session_id] = session
			
			# Add creator as first user
			creator_presence = UserPresence(
				user_id=creator_user_id,
				session_id=session_id,
				name=config.get("creator_name", "User"),
				email=config.get("creator_email", ""),
				permissions=["read", "write", "admin"]
			)
			
			await session.add_user(creator_presence)
			
			# Track user sessions
			if creator_user_id not in self.user_sessions:
				self.user_sessions[creator_user_id] = set()
			self.user_sessions[creator_user_id].add(session_id)
			
			# Store in database
			await self._store_session_info(session)
			
			return session_id
			
		except Exception as e:
			print(f"Create collaboration session error: {e}")
			raise
	
	async def join_collaboration_session(self, session_id: str, user_id: str, user_info: Dict[str, Any]) -> bool:
		"""Join existing collaboration session"""
		try:
			session = self.sessions.get(session_id)
			if not session:
				return False
			
			# Check if user is already in session
			if user_id in session.users:
				return True
			
			# Create user presence
			user_presence = UserPresence(
				user_id=user_id,
				session_id=session_id,
				name=user_info.get("name", "User"),
				email=user_info.get("email", ""),
				permissions=user_info.get("permissions", ["read"])
			)
			
			# Add user to session
			success = await session.add_user(user_presence)
			
			if success:
				# Track user sessions
				if user_id not in self.user_sessions:
					self.user_sessions[user_id] = set()
				self.user_sessions[user_id].add(session_id)
			
			return success
			
		except Exception as e:
			print(f"Join collaboration session error: {e}")
			return False
	
	async def leave_collaboration_session(self, session_id: str, user_id: str) -> bool:
		"""Leave collaboration session"""
		try:
			session = self.sessions.get(session_id)
			if not session:
				return False
			
			# Remove user from session
			success = await session.remove_user(user_id)
			
			if success:
				# Update user session tracking
				if user_id in self.user_sessions:
					self.user_sessions[user_id].discard(session_id)
					if not self.user_sessions[user_id]:
						del self.user_sessions[user_id]
			
			# Clean up session if empty
			if not session.users:
				await self._cleanup_session(session_id)
			
			return success
			
		except Exception as e:
			print(f"Leave collaboration session error: {e}")
			return False
	
	async def update_user_presence(self, session_id: str, user_id: str, presence_data: Dict[str, Any]) -> bool:
		"""Update user presence in session"""
		try:
			session = self.sessions.get(session_id)
			if not session:
				return False
			
			return await session.update_user_presence(user_id, presence_data)
			
		except Exception as e:
			print(f"Update user presence error: {e}")
			return False
	
	async def apply_collaborative_change(self, session_id: str, user_id: str, operation_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
		"""Apply collaborative change operation"""
		try:
			session = self.sessions.get(session_id)
			if not session:
				return False, ["Session not found"]
			
			# Check user permissions
			user = session.users.get(user_id)
			if not user or "write" not in user.permissions:
				return False, ["Insufficient permissions"]
			
			# Create operation
			operation = ChangeOperation(
				user_id=user_id,
				session_id=session_id,
				operation_type=operation_data["operation_type"],
				target_type=operation_data["target_type"],
				target_id=operation_data["target_id"],
				changes=operation_data.get("changes", {}),
				previous_state=operation_data.get("previous_state")
			)
			
			# Apply operation
			success, conflicts = await session.apply_operation(operation)
			
			return success, conflicts
			
		except Exception as e:
			print(f"Apply collaborative change error: {e}")
			return False, [str(e)]
	
	async def start_voice_channel(self, session_id: str, user_id: str, config: Dict[str, Any]) -> str:
		"""Start voice communication channel"""
		try:
			channel_id = uuid7str()
			
			voice_channel = {
				"id": channel_id,
				"session_id": session_id,
				"creator_id": user_id,
				"participants": [user_id],
				"config": config,
				"created_at": datetime.utcnow(),
				"active": True
			}
			
			self.voice_channels[channel_id] = voice_channel
			
			# Broadcast voice channel availability
			session = self.sessions.get(session_id)
			if session:
				await session._broadcast_event("voice_channel_started", {
					"channel_id": channel_id,
					"creator_id": user_id
				})
			
			return channel_id
			
		except Exception as e:
			print(f"Start voice channel error: {e}")
			raise
	
	async def start_screen_sharing(self, session_id: str, user_id: str, config: Dict[str, Any]) -> str:
		"""Start screen sharing session"""
		try:
			sharing_id = uuid7str()
			
			screen_sharing = {
				"id": sharing_id,
				"session_id": session_id,
				"presenter_id": user_id,
				"viewers": [],
				"config": config,
				"started_at": datetime.utcnow(),
				"active": True
			}
			
			self.screen_sharing_sessions[sharing_id] = screen_sharing
			
			# Broadcast screen sharing availability
			session = self.sessions.get(session_id)
			if session:
				await session._broadcast_event("screen_sharing_started", {
					"sharing_id": sharing_id,
					"presenter_id": user_id
				})
			
			return sharing_id
			
		except Exception as e:
			print(f"Start screen sharing error: {e}")
			raise
	
	async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
		"""Get collaboration session status"""
		try:
			session = self.sessions.get(session_id)
			if not session:
				return None
			
			# Calculate session metrics
			total_operations = len(session.operation_history)
			active_users = len([user for user in session.users.values() if user.status == PresenceStatus.ONLINE])
			
			return {
				"session_id": session_id,
				"workflow_id": session.workflow_id,
				"mode": session.mode.value,
				"created_at": session.created_at.isoformat(),
				"last_activity": session.last_activity.isoformat(),
				"users": [
					{
						"user_id": user.user_id,
						"name": user.name,
						"status": user.status.value,
						"joined_at": user.joined_at.isoformat(),
						"last_activity": user.last_activity.isoformat(),
						"permissions": user.permissions
					}
					for user in session.users.values()
				],
				"metrics": {
					"total_operations": total_operations,
					"active_users": active_users,
					"conflict_count": session.conflict_count,
					"operation_count": session.operation_count,
					"sync_latency_ms": session.sync_latency_ms
				},
				"ai_assistant": {
					"enabled": session.ai_assistant_enabled,
					"suggestions_count": len(session.ai_suggestions)
				}
			}
			
		except Exception as e:
			print(f"Get session status error: {e}")
			return None
	
	async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
		"""Get all sessions for a user"""
		try:
			user_session_ids = self.user_sessions.get(user_id, set())
			sessions = []
			
			for session_id in user_session_ids:
				session_status = await self.get_session_status(session_id)
				if session_status:
					sessions.append(session_status)
			
			return sessions
			
		except Exception as e:
			print(f"Get user sessions error: {e}")
			return []
	
	async def _store_session_info(self, session: CollaborationSession):
		"""Store session information in database"""
		try:
			async with get_async_db_session() as session_db:
				from sqlalchemy import text
				
				await session_db.execute(
					text("""
					INSERT INTO workflow_collaboration_sessions (
						session_id, workflow_id, mode, conflict_resolution,
						ai_assistant_enabled, created_at, last_activity
					) VALUES (
						:session_id, :workflow_id, :mode, :conflict_resolution,
						:ai_assistant_enabled, :created_at, :last_activity
					)
					"""),
					{
						"session_id": session.session_id,
						"workflow_id": session.workflow_id,
						"mode": session.mode.value,
						"conflict_resolution": session.conflict_resolution.value,
						"ai_assistant_enabled": session.ai_assistant_enabled,
						"created_at": session.created_at,
						"last_activity": session.last_activity
					}
				)
				await session_db.commit()
				
		except Exception as e:
			print(f"Store session info error: {e}")
	
	async def _cleanup_session(self, session_id: str):
		"""Clean up empty session"""
		try:
			if session_id in self.sessions:
				del self.sessions[session_id]
			
			# Clean up related resources
			voice_channels_to_remove = [
				channel_id for channel_id, channel in self.voice_channels.items()
				if channel["session_id"] == session_id
			]
			
			for channel_id in voice_channels_to_remove:
				del self.voice_channels[channel_id]
			
			sharing_sessions_to_remove = [
				sharing_id for sharing_id, sharing in self.screen_sharing_sessions.items()
				if sharing["session_id"] == session_id
			]
			
			for sharing_id in sharing_sessions_to_remove:
				del self.screen_sharing_sessions[sharing_id]
			
		except Exception as e:
			print(f"Cleanup session error: {e}")


# Global advanced collaboration manager instance
advanced_collaboration = AdvancedCollaborationManager()
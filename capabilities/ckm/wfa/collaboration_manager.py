"""
APG Workflow & Business Process Management - Advanced Collaboration Manager

Real-time process collaboration with APG integration, concurrent editing,
change tracking, and intelligent conflict resolution.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Collaboration Core Classes
# =============================================================================

class CollaborationRole(str, Enum):
	"""Process collaboration roles."""
	PROCESS_OWNER = "process_owner"
	PROCESS_CONTRIBUTOR = "process_contributor"
	PROCESS_REVIEWER = "process_reviewer"
	PROCESS_OBSERVER = "process_observer"
	TASK_COLLABORATOR = "task_collaborator"


class ChangeType(str, Enum):
	"""Types of process changes."""
	ELEMENT_ADDED = "element_added"
	ELEMENT_REMOVED = "element_removed"
	ELEMENT_MODIFIED = "element_modified"
	ELEMENT_MOVED = "element_moved"
	CONNECTION_ADDED = "connection_added"
	CONNECTION_REMOVED = "connection_removed"
	PROPERTY_CHANGED = "property_changed"
	COMMENT_ADDED = "comment_added"
	APPROVAL_REQUEST = "approval_request"
	APPROVAL_GRANTED = "approval_granted"
	APPROVAL_REJECTED = "approval_rejected"


class ConflictResolutionStrategy(str, Enum):
	"""Conflict resolution strategies."""
	LAST_WRITE_WINS = "last_write_wins"
	MERGE_CHANGES = "merge_changes"
	MANUAL_RESOLUTION = "manual_resolution"
	ROLLBACK_CONFLICTING = "rollback_conflicting"


class NotificationType(str, Enum):
	"""Collaboration notification types."""
	USER_JOINED = "user_joined"
	USER_LEFT = "user_left"
	CHANGE_MADE = "change_made"
	COMMENT_ADDED = "comment_added"
	APPROVAL_REQUESTED = "approval_requested"
	CONFLICT_DETECTED = "conflict_detected"
	PROCESS_PUBLISHED = "process_published"
	MENTION_RECEIVED = "mention_received"


@dataclass
class ProcessChange:
	"""Individual process change record."""
	change_id: str = field(default_factory=lambda: f"change_{uuid.uuid4().hex}")
	process_id: str = ""
	user_id: str = ""
	change_type: ChangeType = ChangeType.ELEMENT_MODIFIED
	element_id: Optional[str] = None
	old_value: Optional[Dict[str, Any]] = None
	new_value: Optional[Dict[str, Any]] = None
	change_description: str = ""
	timestamp: datetime = field(default_factory=datetime.utcnow)
	session_id: Optional[str] = None
	tenant_id: str = ""
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization."""
		return {
			'change_id': self.change_id,
			'process_id': self.process_id,
			'user_id': self.user_id,
			'change_type': self.change_type.value,
			'element_id': self.element_id,
			'old_value': self.old_value,
			'new_value': self.new_value,
			'change_description': self.change_description,
			'timestamp': self.timestamp.isoformat(),
			'session_id': self.session_id,
			'tenant_id': self.tenant_id
		}


@dataclass
class ProcessComment:
	"""Process comment with threading and mentions."""
	comment_id: str = field(default_factory=lambda: f"comment_{uuid.uuid4().hex}")
	process_id: str = ""
	element_id: Optional[str] = None
	user_id: str = ""
	content: str = ""
	parent_comment_id: Optional[str] = None
	mentions: List[str] = field(default_factory=list)
	attachments: List[Dict[str, Any]] = field(default_factory=list)
	is_resolved: bool = False
	resolved_by: Optional[str] = None
	resolved_at: Optional[datetime] = None
	timestamp: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization."""
		return {
			'comment_id': self.comment_id,
			'process_id': self.process_id,
			'element_id': self.element_id,
			'user_id': self.user_id,
			'content': self.content,
			'parent_comment_id': self.parent_comment_id,
			'mentions': self.mentions,
			'attachments': self.attachments,
			'is_resolved': self.is_resolved,
			'resolved_by': self.resolved_by,
			'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
			'timestamp': self.timestamp.isoformat(),
			'tenant_id': self.tenant_id
		}


@dataclass
class CollaborationConflict:
	"""Detected collaboration conflict."""
	conflict_id: str = field(default_factory=lambda: f"conflict_{uuid.uuid4().hex}")
	process_id: str = ""
	conflicting_changes: List[ProcessChange] = field(default_factory=list)
	conflict_description: str = ""
	suggested_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.MANUAL_RESOLUTION
	is_resolved: bool = False
	resolution_strategy: Optional[ConflictResolutionStrategy] = None
	resolved_by: Optional[str] = None
	resolved_at: Optional[datetime] = None
	created_at: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""


@dataclass
class ProcessApproval:
	"""Process approval workflow."""
	approval_id: str = field(default_factory=lambda: f"approval_{uuid.uuid4().hex}")
	process_id: str = ""
	requested_by: str = ""
	approvers: List[str] = field(default_factory=list)
	approval_type: str = "publish"  # publish, deploy, major_change
	description: str = ""
	required_approvals: int = 1
	received_approvals: List[Dict[str, Any]] = field(default_factory=list)
	status: str = "pending"  # pending, approved, rejected, cancelled
	created_at: datetime = field(default_factory=datetime.utcnow)
	deadline: Optional[datetime] = None
	tenant_id: str = ""


@dataclass
class CollaborationSession:
	"""Active collaboration session."""
	session_id: str
	process_id: str
	tenant_id: str
	owner_id: str
	participants: Dict[str, CollaborationRole] = field(default_factory=dict)
	active_users: Set[str] = field(default_factory=set)
	change_buffer: List[ProcessChange] = field(default_factory=list)
	cursor_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	session_settings: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Change Detection and Tracking
# =============================================================================

class ChangeTracker:
	"""Track and analyze process changes for collaboration."""
	
	def __init__(self):
		self.change_history: Dict[str, List[ProcessChange]] = defaultdict(list)
		self.conflict_detector = ConflictDetector()
	
	async def track_change(
		self,
		change: ProcessChange,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Track a new process change."""
		try:
			# Validate change
			if not await self._validate_change(change, context):
				return WBPMServiceResponse(
					success=False,
					message="Invalid change data",
					errors=["Change validation failed"]
				)
			
			# Store change
			self.change_history[change.process_id].append(change)
			
			# Keep only recent changes (last 1000 per process)
			if len(self.change_history[change.process_id]) > 1000:
				self.change_history[change.process_id] = self.change_history[change.process_id][-1000:]
			
			# Detect conflicts
			conflicts = await self.conflict_detector.detect_conflicts(
				change, self.change_history[change.process_id]
			)
			
			logger.info(f"Change tracked: {change.change_id} for process {change.process_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Change tracked successfully",
				data={
					"change_id": change.change_id,
					"conflicts_detected": len(conflicts) > 0,
					"conflicts": [conflict.conflict_id for conflict in conflicts]
				}
			)
			
		except Exception as e:
			logger.error(f"Error tracking change: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to track change: {e}",
				errors=[str(e)]
			)
	
	async def get_change_history(
		self,
		process_id: str,
		context: APGTenantContext,
		limit: int = 100,
		since: Optional[datetime] = None
	) -> WBPMPagedResponse:
		"""Get change history for process."""
		try:
			changes = self.change_history.get(process_id, [])
			
			# Filter by date if specified
			if since:
				changes = [change for change in changes if change.timestamp >= since]
			
			# Sort by timestamp (newest first)
			changes.sort(key=lambda x: x.timestamp, reverse=True)
			
			# Apply limit
			limited_changes = changes[:limit]
			
			# Convert to dict format
			change_data = [change.to_dict() for change in limited_changes]
			
			return WBPMPagedResponse(
				items=change_data,
				total_count=len(changes),
				page=1,
				page_size=limit,
				has_next=len(changes) > limit,
				has_previous=False
			)
			
		except Exception as e:
			logger.error(f"Error getting change history: {e}")
			return WBPMPagedResponse(
				items=[],
				total_count=0,
				page=1,
				page_size=limit,
				has_next=False,
				has_previous=False
			)
	
	async def _validate_change(self, change: ProcessChange, context: APGTenantContext) -> bool:
		"""Validate change data."""
		if not change.process_id or not change.user_id:
			return False
		
		if change.tenant_id != context.tenant_id:
			return False
		
		return True


# =============================================================================
# Conflict Detection and Resolution
# =============================================================================

class ConflictDetector:
	"""Detect and analyze collaboration conflicts."""
	
	def __init__(self):
		self.conflict_rules = self._initialize_conflict_rules()
	
	async def detect_conflicts(
		self,
		new_change: ProcessChange,
		change_history: List[ProcessChange]
	) -> List[CollaborationConflict]:
		"""Detect conflicts with recent changes."""
		conflicts = []
		
		# Look for conflicts in recent changes (last 5 minutes)
		recent_threshold = datetime.utcnow() - timedelta(minutes=5)
		recent_changes = [
			change for change in change_history
			if change.timestamp >= recent_threshold and change.change_id != new_change.change_id
		]
		
		for rule_name, rule_func in self.conflict_rules.items():
			conflict = await rule_func(new_change, recent_changes)
			if conflict:
				conflicts.append(conflict)
		
		return conflicts
	
	async def _detect_simultaneous_element_edit(
		self,
		new_change: ProcessChange,
		recent_changes: List[ProcessChange]
	) -> Optional[CollaborationConflict]:
		"""Detect simultaneous editing of same element."""
		if not new_change.element_id:
			return None
		
		conflicting_changes = [
			change for change in recent_changes
			if (change.element_id == new_change.element_id and
				change.user_id != new_change.user_id and
				change.change_type in [ChangeType.ELEMENT_MODIFIED, ChangeType.PROPERTY_CHANGED])
		]
		
		if conflicting_changes:
			return CollaborationConflict(
				process_id=new_change.process_id,
				conflicting_changes=[new_change] + conflicting_changes,
				conflict_description=f"Multiple users editing element {new_change.element_id} simultaneously",
				suggested_resolution=ConflictResolutionStrategy.MERGE_CHANGES,
				tenant_id=new_change.tenant_id
			)
		
		return None
	
	async def _detect_element_deletion_conflict(
		self,
		new_change: ProcessChange,
		recent_changes: List[ProcessChange]
	) -> Optional[CollaborationConflict]:
		"""Detect conflicts with deleted elements."""
		if new_change.change_type != ChangeType.ELEMENT_REMOVED:
			return None
		
		# Check if other users are modifying the element being deleted
		conflicting_changes = [
			change for change in recent_changes
			if (change.element_id == new_change.element_id and
				change.user_id != new_change.user_id and
				change.change_type in [ChangeType.ELEMENT_MODIFIED, ChangeType.PROPERTY_CHANGED])
		]
		
		if conflicting_changes:
			return CollaborationConflict(
				process_id=new_change.process_id,
				conflicting_changes=[new_change] + conflicting_changes,
				conflict_description=f"Element {new_change.element_id} deleted while being modified by others",
				suggested_resolution=ConflictResolutionStrategy.MANUAL_RESOLUTION,
				tenant_id=new_change.tenant_id
			)
		
		return None
	
	async def _detect_flow_connection_conflict(
		self,
		new_change: ProcessChange,
		recent_changes: List[ProcessChange]
	) -> Optional[CollaborationConflict]:
		"""Detect conflicts in flow connections."""
		if new_change.change_type not in [ChangeType.CONNECTION_ADDED, ChangeType.CONNECTION_REMOVED]:
			return None
		
		# Check for conflicting connection changes
		conflicting_changes = [
			change for change in recent_changes
			if (change.change_type in [ChangeType.CONNECTION_ADDED, ChangeType.CONNECTION_REMOVED] and
				change.user_id != new_change.user_id)
		]
		
		# Analyze if connections conflict
		for change in conflicting_changes:
			if self._connections_conflict(new_change, change):
				return CollaborationConflict(
					process_id=new_change.process_id,
					conflicting_changes=[new_change, change],
					conflict_description="Conflicting connection modifications detected",
					suggested_resolution=ConflictResolutionStrategy.MANUAL_RESOLUTION,
					tenant_id=new_change.tenant_id
				)
		
		return None
	
	def _connections_conflict(self, change1: ProcessChange, change2: ProcessChange) -> bool:
		"""Check if two connection changes conflict."""
		# Simplified conflict detection - in production would be more sophisticated
		if not change1.new_value or not change2.new_value:
			return False
		
		source1 = change1.new_value.get('source_ref')
		target1 = change1.new_value.get('target_ref')
		source2 = change2.new_value.get('source_ref')
		target2 = change2.new_value.get('target_ref')
		
		# Check if they involve the same source or target
		return (source1 == source2 or target1 == target2 or
				source1 == target2 or target1 == source2)
	
	def _initialize_conflict_rules(self) -> Dict[str, Any]:
		"""Initialize conflict detection rules."""
		return {
			'simultaneous_element_edit': self._detect_simultaneous_element_edit,
			'element_deletion_conflict': self._detect_element_deletion_conflict,
			'flow_connection_conflict': self._detect_flow_connection_conflict
		}


# =============================================================================
# Real-time Collaboration Manager
# =============================================================================

class CollaborationManager:
	"""Manage real-time process collaboration."""
	
	def __init__(self):
		self.active_sessions: Dict[str, CollaborationSession] = {}
		self.change_tracker = ChangeTracker()
		self.comments: Dict[str, List[ProcessComment]] = defaultdict(list)
		self.approvals: Dict[str, List[ProcessApproval]] = defaultdict(list)
		self.conflicts: Dict[str, List[CollaborationConflict]] = defaultdict(list)
	
	async def start_collaboration_session(
		self,
		process_id: str,
		context: APGTenantContext,
		session_settings: Optional[Dict[str, Any]] = None
	) -> WBPMServiceResponse:
		"""Start new collaboration session."""
		try:
			session_id = f"collab_{uuid.uuid4().hex}"
			
			session = CollaborationSession(
				session_id=session_id,
				process_id=process_id,
				tenant_id=context.tenant_id,
				owner_id=context.user_id,
				session_settings=session_settings or {}
			)
			
			# Add owner as participant
			session.participants[context.user_id] = CollaborationRole.PROCESS_OWNER
			session.active_users.add(context.user_id)
			
			self.active_sessions[session_id] = session
			
			logger.info(f"Collaboration session started: {session_id} for process {process_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Collaboration session started successfully",
				data={
					"session_id": session_id,
					"process_id": process_id,
					"owner_id": context.user_id,
					"session_settings": session.session_settings
				}
			)
			
		except Exception as e:
			logger.error(f"Error starting collaboration session: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to start collaboration session: {e}",
				errors=[str(e)]
			)
	
	async def join_collaboration_session(
		self,
		session_id: str,
		context: APGTenantContext,
		requested_role: Optional[CollaborationRole] = None
	) -> WBPMServiceResponse:
		"""Join existing collaboration session."""
		try:
			session = self.active_sessions.get(session_id)
			if not session:
				return WBPMServiceResponse(
					success=False,
					message="Collaboration session not found",
					errors=["Session not found"]
				)
			
			# Verify tenant access
			if session.tenant_id != context.tenant_id:
				return WBPMServiceResponse(
					success=False,
					message="Access denied to collaboration session",
					errors=["Tenant access denied"]
				)
			
			# Determine role
			role = requested_role or CollaborationRole.PROCESS_OBSERVER
			
			# Add participant
			session.participants[context.user_id] = role
			session.active_users.add(context.user_id)
			session.last_activity = datetime.utcnow()
			
			# Notify other participants
			await self._notify_session_participants(session_id, {
				"type": NotificationType.USER_JOINED,
				"user_id": context.user_id,
				"role": role.value,
				"timestamp": datetime.utcnow().isoformat()
			})
			
			return WBPMServiceResponse(
				success=True,
				message="Joined collaboration session successfully",
				data={
					"session_id": session_id,
					"role": role.value,
					"participants": {uid: role.value for uid, role in session.participants.items()},
					"active_users": list(session.active_users)
				}
			)
			
		except Exception as e:
			logger.error(f"Error joining collaboration session {session_id}: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to join collaboration session: {e}",
				errors=[str(e)]
			)
	
	async def broadcast_change(
		self,
		session_id: str,
		change: ProcessChange,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Broadcast change to all session participants."""
		try:
			session = self.active_sessions.get(session_id)
			if not session:
				return WBPMServiceResponse(
					success=False,
					message="Collaboration session not found",
					errors=["Session not found"]
				)
			
			# Verify user is participant
			if context.user_id not in session.participants:
				return WBPMServiceResponse(
					success=False,
					message="Not a participant in this session",
					errors=["Access denied"]
				)
			
			# Track change
			change.session_id = session_id
			track_result = await self.change_tracker.track_change(change, context)
			
			if not track_result.success:
				return track_result
			
			# Add to session buffer
			session.change_buffer.append(change)
			session.last_activity = datetime.utcnow()
			
			# Notify participants
			await self._notify_session_participants(session_id, {
				"type": NotificationType.CHANGE_MADE,
				"change": change.to_dict(),
				"conflicts": track_result.data.get("conflicts", []),
				"timestamp": datetime.utcnow().isoformat()
			})
			
			return WBPMServiceResponse(
				success=True,
				message="Change broadcasted successfully",
				data={
					"change_id": change.change_id,
					"participants_notified": len(session.active_users) - 1
				}
			)
			
		except Exception as e:
			logger.error(f"Error broadcasting change: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to broadcast change: {e}",
				errors=[str(e)]
			)
	
	async def add_comment(
		self,
		comment: ProcessComment,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Add comment to process."""
		try:
			# Store comment
			self.comments[comment.process_id].append(comment)
			
			# Process mentions
			mention_notifications = []
			for mentioned_user in comment.mentions:
				mention_notifications.append({
					"type": NotificationType.MENTION_RECEIVED,
					"mentioned_by": context.user_id,
					"comment_id": comment.comment_id,
					"content": comment.content[:100] + "..." if len(comment.content) > 100 else comment.content,
					"timestamp": datetime.utcnow().isoformat()
				})
			
			# Send mention notifications
			for notification in mention_notifications:
				await self._send_mention_notification(notification)
			
			# Notify session participants if in active session
			session = self._find_active_session_for_process(comment.process_id)
			if session:
				await self._notify_session_participants(session.session_id, {
					"type": NotificationType.COMMENT_ADDED,
					"comment": comment.to_dict(),
					"timestamp": datetime.utcnow().isoformat()
				})
			
			logger.info(f"Comment added: {comment.comment_id} to process {comment.process_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Comment added successfully",
				data={
					"comment_id": comment.comment_id,
					"mentions_sent": len(mention_notifications)
				}
			)
			
		except Exception as e:
			logger.error(f"Error adding comment: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to add comment: {e}",
				errors=[str(e)]
			)
	
	async def request_approval(
		self,
		approval: ProcessApproval,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Request process approval."""
		try:
			# Store approval request
			self.approvals[approval.process_id].append(approval)
			
			# Notify approvers
			for approver_id in approval.approvers:
				await self._send_approval_notification(approver_id, approval)
			
			# Notify session participants
			session = self._find_active_session_for_process(approval.process_id)
			if session:
				await self._notify_session_participants(session.session_id, {
					"type": NotificationType.APPROVAL_REQUESTED,
					"approval": {
						"approval_id": approval.approval_id,
						"type": approval.approval_type,
						"description": approval.description,
						"approvers": approval.approvers,
						"deadline": approval.deadline.isoformat() if approval.deadline else None
					},
					"timestamp": datetime.utcnow().isoformat()
				})
			
			logger.info(f"Approval requested: {approval.approval_id} for process {approval.process_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Approval requested successfully",
				data={
					"approval_id": approval.approval_id,
					"approvers_notified": len(approval.approvers)
				}
			)
			
		except Exception as e:
			logger.error(f"Error requesting approval: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to request approval: {e}",
				errors=[str(e)]
			)
	
	async def update_cursor_position(
		self,
		session_id: str,
		cursor_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Update user cursor position in session."""
		try:
			session = self.active_sessions.get(session_id)
			if not session:
				return WBPMServiceResponse(
					success=False,
					message="Collaboration session not found",
					errors=["Session not found"]
				)
			
			# Update cursor position
			session.cursor_positions[context.user_id] = {
				"x": cursor_data.get("x", 0),
				"y": cursor_data.get("y", 0),
				"element_id": cursor_data.get("element_id"),
				"timestamp": datetime.utcnow().isoformat()
			}
			
			# Broadcast cursor position to other participants
			await self._notify_session_participants(session_id, {
				"type": "cursor_updated",
				"user_id": context.user_id,
				"cursor": session.cursor_positions[context.user_id]
			}, exclude_user=context.user_id)
			
			return WBPMServiceResponse(
				success=True,
				message="Cursor position updated successfully",
				data={"updated": True}
			)
			
		except Exception as e:
			logger.error(f"Error updating cursor position: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to update cursor position: {e}",
				errors=[str(e)]
			)
	
	async def get_collaboration_status(
		self,
		process_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Get collaboration status for process."""
		try:
			# Find active session
			session = self._find_active_session_for_process(process_id)
			
			# Get recent comments
			recent_comments = self.comments.get(process_id, [])[-10:]  # Last 10 comments
			
			# Get pending approvals
			pending_approvals = [
				approval for approval in self.approvals.get(process_id, [])
				if approval.status == "pending"
			]
			
			# Get unresolved conflicts
			unresolved_conflicts = [
				conflict for conflict in self.conflicts.get(process_id, [])
				if not conflict.is_resolved
			]
			
			# Get recent changes
			change_history = await self.change_tracker.get_change_history(
				process_id, context, limit=20
			)
			
			return WBPMServiceResponse(
				success=True,
				message="Collaboration status retrieved successfully",
				data={
					"process_id": process_id,
					"active_session": {
						"session_id": session.session_id,
						"participants": {uid: role.value for uid, role in session.participants.items()},
						"active_users": list(session.active_users),
						"cursor_positions": session.cursor_positions
					} if session else None,
					"recent_comments": [comment.to_dict() for comment in recent_comments],
					"pending_approvals": len(pending_approvals),
					"unresolved_conflicts": len(unresolved_conflicts),
					"recent_changes": change_history.items[-5:] if change_history.items else []
				}
			)
			
		except Exception as e:
			logger.error(f"Error getting collaboration status: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get collaboration status: {e}",
				errors=[str(e)]
			)
	
	def _find_active_session_for_process(self, process_id: str) -> Optional[CollaborationSession]:
		"""Find active collaboration session for process."""
		for session in self.active_sessions.values():
			if session.process_id == process_id:
				return session
		return None
	
	async def _notify_session_participants(
		self,
		session_id: str,
		notification: Dict[str, Any],
		exclude_user: Optional[str] = None
	) -> None:
		"""Notify all session participants."""
		session = self.active_sessions.get(session_id)
		if not session:
			return
		
		participants_to_notify = [
			user_id for user_id in session.active_users
			if user_id != exclude_user
		]
		
		# In production, use WebSocket or APG real-time service
		logger.info(f"Notifying {len(participants_to_notify)} participants in session {session_id}: {notification['type']}")
	
	async def _send_mention_notification(self, notification: Dict[str, Any]) -> None:
		"""Send mention notification to user."""
		# In production, integrate with APG notification service
		logger.info(f"Mention notification sent: {notification}")
	
	async def _send_approval_notification(self, approver_id: str, approval: ProcessApproval) -> None:
		"""Send approval request notification."""
		# In production, integrate with APG notification service
		logger.info(f"Approval notification sent to {approver_id} for approval {approval.approval_id}")


# =============================================================================
# Service Factory
# =============================================================================

def create_collaboration_manager() -> CollaborationManager:
	"""Create and configure collaboration manager."""
	manager = CollaborationManager()
	logger.info("Collaboration manager created and configured")
	return manager


# Export main classes
__all__ = [
	'CollaborationManager',
	'ChangeTracker',
	'ConflictDetector',
	'ProcessChange',
	'ProcessComment',
	'CollaborationConflict',
	'ProcessApproval',
	'CollaborationSession',
	'CollaborationRole',
	'ChangeType',
	'ConflictResolutionStrategy',
	'NotificationType',
	'create_collaboration_manager'
]
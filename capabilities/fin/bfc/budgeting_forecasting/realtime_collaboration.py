"""
APG Budgeting & Forecasting - Real-Time Collaboration

Enterprise-grade real-time collaborative budget editing with conflict resolution,
live comments, change tracking, and seamless APG platform integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from decimal import Decimal
from uuid import UUID
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import hashlib

import asyncpg
from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict

from .models import (
	APGBaseModel, BFBudgetStatus, BFLineType, BFApprovalStatus,
	PositiveAmount, CurrencyCode, NonEmptyString
)
from .service import APGTenantContext, BFServiceConfig, ServiceResponse, APGServiceBase
from uuid_extensions import uuid7str


# =============================================================================
# Real-Time Collaboration Models
# =============================================================================

class CollaborationEventType(str, Enum):
	"""Real-time collaboration event types."""
	USER_JOIN = "user_join"
	USER_LEAVE = "user_leave"
	CELL_EDIT = "cell_edit"
	CELL_SELECT = "cell_select"
	COMMENT_ADD = "comment_add"
	COMMENT_REPLY = "comment_reply"
	COMMENT_RESOLVE = "comment_resolve"
	CHANGE_SUBMIT = "change_submit"
	CHANGE_APPROVE = "change_approve"
	CHANGE_REJECT = "change_reject"
	CONFLICT_DETECTED = "conflict_detected"
	CONFLICT_RESOLVED = "conflict_resolved"
	SESSION_LOCK = "session_lock"
	SESSION_UNLOCK = "session_unlock"


class UserPresenceStatus(str, Enum):
	"""User presence status in collaboration session."""
	ACTIVE = "active"
	IDLE = "idle"
	EDITING = "editing"
	REVIEWING = "reviewing"
	AWAY = "away"
	DISCONNECTED = "disconnected"


class ConflictResolutionStrategy(str, Enum):
	"""Conflict resolution strategies."""
	LAST_WRITER_WINS = "last_writer_wins"
	FIRST_WRITER_WINS = "first_writer_wins"
	MANUAL_RESOLUTION = "manual_resolution"
	MERGE_CHANGES = "merge_changes"
	ESCALATE_TO_SUPERVISOR = "escalate_to_supervisor"


class CollaborationSession(APGBaseModel):
	"""Real-time collaboration session model."""
	
	session_name: NonEmptyString = Field(..., max_length=255)
	budget_id: str = Field(...)
	session_owner: str = Field(...)
	
	# Session configuration
	max_participants: int = Field(default=10, ge=1, le=50)
	allow_guest_users: bool = Field(default=False)
	require_approval_for_changes: bool = Field(default=True)
	auto_save_interval: int = Field(default=30, ge=5, le=300)  # seconds
	
	# Session state
	is_active: bool = Field(default=True)
	session_started_at: datetime = Field(default_factory=datetime.utcnow)
	session_ends_at: Optional[datetime] = Field(None)
	last_activity_at: datetime = Field(default_factory=datetime.utcnow)
	
	# Collaboration settings
	conflict_resolution_strategy: ConflictResolutionStrategy = Field(default=ConflictResolutionStrategy.MANUAL_RESOLUTION)
	enable_live_cursors: bool = Field(default=True)
	enable_live_selections: bool = Field(default=True)
	enable_change_notifications: bool = Field(default=True)
	
	# Permissions and access
	session_permissions: Dict[str, Any] = Field(default_factory=dict)
	allowed_participants: List[str] = Field(default_factory=list)
	blocked_participants: List[str] = Field(default_factory=list)
	
	# Session metadata
	session_version: int = Field(default=1, ge=1)
	total_changes_made: int = Field(default=0, ge=0)
	total_comments_added: int = Field(default=0, ge=0)
	
	# APG Integration
	real_time_room_id: Optional[str] = Field(None, description="APG real_time_collaboration room ID")
	notification_channel_id: Optional[str] = Field(None, description="APG notification channel")


class CollaborationParticipant(BaseModel):
	"""Collaboration session participant model."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	participant_id: str = Field(default_factory=uuid7str)
	session_id: str = Field(...)
	user_id: str = Field(...)
	user_name: NonEmptyString = Field(..., max_length=100)
	user_email: Optional[str] = Field(None)
	
	# User role and permissions
	role: str = Field(..., max_length=50)  # owner, editor, reviewer, viewer
	permissions: List[str] = Field(default_factory=list)
	can_edit_budget: bool = Field(default=True)
	can_add_comments: bool = Field(default=True)
	can_approve_changes: bool = Field(default=False)
	
	# Presence and activity
	presence_status: UserPresenceStatus = Field(default=UserPresenceStatus.ACTIVE)
	joined_at: datetime = Field(default_factory=datetime.utcnow)
	last_seen_at: datetime = Field(default_factory=datetime.utcnow)
	last_activity_at: datetime = Field(default_factory=datetime.utcnow)
	
	# Current editing context
	current_section: Optional[str] = Field(None)  # budget_header, line_items, totals
	current_line_id: Optional[str] = Field(None)
	current_field: Optional[str] = Field(None)
	cursor_position: Optional[Dict[str, Any]] = Field(None)
	
	# Session statistics
	changes_made: int = Field(default=0, ge=0)
	comments_added: int = Field(default=0, ge=0)
	session_duration_minutes: int = Field(default=0, ge=0)
	
	# Connection details
	connection_id: Optional[str] = Field(None)
	ip_address: Optional[str] = Field(None)
	user_agent: Optional[str] = Field(None)


class CollaborationEvent(BaseModel):
	"""Real-time collaboration event model."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	event_id: str = Field(default_factory=uuid7str)
	session_id: str = Field(...)
	event_type: CollaborationEventType = Field(...)
	
	# Event source
	user_id: str = Field(...)
	user_name: str = Field(...)
	participant_id: str = Field(...)
	
	# Event target
	target_type: str = Field(..., max_length=50)  # budget, line_item, comment, session
	target_id: Optional[str] = Field(None)
	target_field: Optional[str] = Field(None)
	
	# Event data
	event_data: Dict[str, Any] = Field(default_factory=dict)
	previous_value: Optional[Any] = Field(None)
	new_value: Optional[Any] = Field(None)
	
	# Change validation
	is_valid_change: bool = Field(default=True)
	validation_errors: List[str] = Field(default_factory=list)
	requires_approval: bool = Field(default=False)
	
	# Conflict detection
	has_conflict: bool = Field(default=False)
	conflict_with_event: Optional[str] = Field(None)
	conflict_resolution: Optional[str] = Field(None)
	
	# Event metadata
	event_timestamp: datetime = Field(default_factory=datetime.utcnow)
	event_sequence: int = Field(..., ge=1)
	event_checksum: Optional[str] = Field(None)
	
	# Delivery tracking
	delivered_to: List[str] = Field(default_factory=list)
	acknowledged_by: List[str] = Field(default_factory=list)
	failed_deliveries: List[str] = Field(default_factory=list)


class BudgetComment(APGBaseModel):
	"""Budget comment with threading and resolution support."""
	
	comment_text: str = Field(..., min_length=1, max_length=2000)
	comment_type: str = Field(default="general", max_length=50)  # general, suggestion, issue, question
	
	# Comment targeting
	budget_id: str = Field(...)
	line_item_id: Optional[str] = Field(None)
	field_name: Optional[str] = Field(None)
	
	# Comment threading
	parent_comment_id: Optional[str] = Field(None)
	thread_id: str = Field(...)
	is_root_comment: bool = Field(default=True)
	
	# Comment metadata
	author_name: str = Field(...)
	author_role: Optional[str] = Field(None)
	mentions: List[str] = Field(default_factory=list)  # @user mentions
	tags: List[str] = Field(default_factory=list)
	
	# Comment status
	is_resolved: bool = Field(default=False)
	resolved_by: Optional[str] = Field(None)
	resolved_at: Optional[datetime] = Field(None)
	resolution_note: Optional[str] = Field(None)
	
	# Reactions and engagement
	reactions: Dict[str, List[str]] = Field(default_factory=dict)  # emoji -> [user_ids]
	reply_count: int = Field(default=0, ge=0)
	
	# Positioning (for UI)
	position_x: Optional[float] = Field(None)
	position_y: Optional[float] = Field(None)
	anchor_element: Optional[str] = Field(None)


class ChangeRequest(APGBaseModel):
	"""Change request for collaborative budget editing."""
	
	change_title: NonEmptyString = Field(..., max_length=255)
	change_description: Optional[str] = Field(None, max_length=1000)
	
	# Change details
	budget_id: str = Field(...)
	session_id: str = Field(...)
	requested_by: str = Field(...)
	
	# Change specification
	changes: List[Dict[str, Any]] = Field(...)
	affected_line_items: List[str] = Field(default_factory=list)
	impact_analysis: Dict[str, Any] = Field(default_factory=dict)
	
	# Approval workflow
	requires_approval: bool = Field(default=True)
	approval_level_required: int = Field(default=1, ge=0, le=5)
	approvers: List[str] = Field(default_factory=list)
	approved_by: List[str] = Field(default_factory=list)
	rejected_by: List[str] = Field(default_factory=list)
	
	# Change status
	status: str = Field(default="pending", max_length=20)  # pending, approved, rejected, applied
	applied_at: Optional[datetime] = Field(None)
	applied_by: Optional[str] = Field(None)
	
	# Conflict resolution
	has_conflicts: bool = Field(default=False)
	conflict_details: List[Dict[str, Any]] = Field(default_factory=list)
	conflict_resolution_strategy: Optional[ConflictResolutionStrategy] = Field(None)
	
	# Change validation
	validation_passed: bool = Field(default=False)
	validation_errors: List[str] = Field(default_factory=list)
	business_impact_score: Optional[float] = Field(None, ge=0.0, le=10.0)


# =============================================================================
# Real-Time Collaboration Service
# =============================================================================

class RealTimeCollaborationService(APGServiceBase):
	"""
	Comprehensive real-time collaboration service for budget planning
	with live editing, conflict resolution, and APG platform integration.
	"""
	
	def __init__(self, context: APGTenantContext, config: BFServiceConfig):
		super().__init__(context, config)
		self._active_sessions: Dict[str, CollaborationSession] = {}
		self._session_participants: Dict[str, List[CollaborationParticipant]] = {}
		self._event_handlers: Dict[CollaborationEventType, List[Callable]] = {}
		self._conflict_resolvers: Dict[ConflictResolutionStrategy, Callable] = {}
		
		# Initialize event handlers
		self._initialize_event_handlers()
		self._initialize_conflict_resolvers()
	
	async def create_collaboration_session(self, session_config: Dict[str, Any]) -> ServiceResponse:
		"""Create a new real-time collaboration session."""
		try:
			# Validate permissions
			if not await self._validate_permissions('collaboration.create_session'):
				raise PermissionError("Insufficient permissions to create collaboration session")
			
			# Inject context data
			session_config.update({
				'tenant_id': self.context.tenant_id,
				'session_owner': self.context.user_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			})
			
			# Create session model
			session = CollaborationSession(**session_config)
			
			# Validate session configuration
			validation_result = await self._validate_session_config(session)
			if not validation_result['is_valid']:
				return ServiceResponse(
					success=False,
					message="Session configuration validation failed",
					errors=validation_result['errors']
				)
			
			# Check budget access permissions
			budget_access = await self._validate_budget_access(session.budget_id)
			if not budget_access:
				raise PermissionError("No access to specified budget")
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert session
				session_id = await self._insert_collaboration_session(session)
				
				# Create APG real-time room
				room_id = await self._create_real_time_room(session_id, session)
				await self._update_session_room_id(session_id, room_id)
				
				# Add session owner as first participant
				owner_participant = await self._add_session_participant(
					session_id, self.context.user_id, 'owner', {}
				)
				
				# Initialize session state
				await self._initialize_session_state(session_id, session)
				
				# Set up event streaming
				await self._setup_event_streaming(session_id, room_id)
				
				# Store in active sessions cache
				self._active_sessions[session_id] = session
				self._session_participants[session_id] = [owner_participant]
				
				# Audit session creation
				await self._audit_action('create_session', 'collaboration', session_id,
										new_data={'budget_id': session.budget_id})
			
			return ServiceResponse(
				success=True,
				message=f"Collaboration session '{session.session_name}' created successfully",
				data={
					'session_id': session_id,
					'room_id': room_id,
					'participant_id': owner_participant.participant_id,
					'session_url': f"/collaborate/{session_id}",
					'real_time_endpoint': f"wss://realtime.apg.platform/rooms/{room_id}"
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_collaboration_session')
	
	async def join_collaboration_session(self, session_id: str, join_config: Dict[str, Any]) -> ServiceResponse:
		"""Join an existing collaboration session."""
		try:
			# Validate permissions
			if not await self._validate_permissions('collaboration.join_session', session_id):
				raise PermissionError("Insufficient permissions to join collaboration session")
			
			# Get session
			session = await self._get_collaboration_session(session_id)
			if not session:
				raise ValueError("Collaboration session not found")
			
			# Check if session is active
			if not session['is_active']:
				raise ValueError("Collaboration session is not active")
			
			# Check participant limits
			current_participants = await self._get_session_participant_count(session_id)
			if current_participants >= session['max_participants']:
				raise ValueError("Session has reached maximum participant limit")
			
			# Check if user is allowed to join
			is_allowed = await self._check_participant_access(session_id, self.context.user_id, session)
			if not is_allowed:
				raise PermissionError("Not authorized to join this session")
			
			# Check if user is already in session
			existing_participant = await self._get_user_participant(session_id, self.context.user_id)
			if existing_participant:
				# Rejoin existing participant
				participant_id = await self._rejoin_session_participant(session_id, existing_participant['participant_id'])
				
				# Broadcast rejoin event
				await self._broadcast_collaboration_event(session_id, CollaborationEventType.USER_JOIN, {
					'user_id': self.context.user_id,
					'user_name': join_config.get('user_name', 'Unknown User'),
					'rejoined': True
				})
				
				return ServiceResponse(
					success=True,
					message="Rejoined collaboration session successfully",
					data={
						'session_id': session_id,
						'participant_id': participant_id,
						'rejoined': True
					}
				)
			
			# Add new participant
			participant_data = {
				'user_name': join_config.get('user_name', 'Unknown User'),
				'user_email': join_config.get('user_email'),
				'role': join_config.get('role', 'editor'),
				'connection_id': join_config.get('connection_id'),
				'ip_address': join_config.get('ip_address'),
				'user_agent': join_config.get('user_agent')
			}
			
			participant = await self._add_session_participant(
				session_id, self.context.user_id, participant_data.get('role', 'editor'), participant_data
			)
			
			# Start database transaction
			async with self._connection.transaction():
				# Update session activity
				await self._update_session_activity(session_id)
				
				# Join APG real-time room
				await self._join_real_time_room(session['real_time_room_id'], self.context.user_id)
				
				# Load session state for new participant
				session_state = await self._load_session_state_for_participant(session_id, participant.participant_id)
				
				# Broadcast join event
				await self._broadcast_collaboration_event(session_id, CollaborationEventType.USER_JOIN, {
					'user_id': self.context.user_id,
					'user_name': participant.user_name,
					'role': participant.role,
					'joined_at': participant.joined_at.isoformat()
				})
				
				# Update cached participants
				if session_id in self._session_participants:
					self._session_participants[session_id].append(participant)
				
				# Audit session join
				await self._audit_action('join_session', 'collaboration', session_id,
										new_data={'participant_id': participant.participant_id})
			
			return ServiceResponse(
				success=True,
				message="Joined collaboration session successfully",
				data={
					'session_id': session_id,
					'participant_id': participant.participant_id,
					'session_state': session_state,
					'other_participants': await self._get_other_participants(session_id, participant.participant_id),
					'room_id': session['real_time_room_id']
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'join_collaboration_session')
	
	async def submit_change_request(self, session_id: str, change_data: Dict[str, Any]) -> ServiceResponse:
		"""Submit a change request for collaborative review."""
		try:
			# Validate permissions
			if not await self._validate_permissions('collaboration.submit_change', session_id):
				raise PermissionError("Insufficient permissions to submit changes")
			
			# Get session and validate
			session = await self._get_collaboration_session(session_id)
			if not session or not session['is_active']:
				raise ValueError("Invalid or inactive collaboration session")
			
			# Get participant
			participant = await self._get_user_participant(session_id, self.context.user_id)
			if not participant or not participant['can_edit_budget']:
				raise PermissionError("No edit permissions in this session")
			
			# Create change request
			change_request_data = {
				**change_data,
				'budget_id': session['budget_id'],
				'session_id': session_id,
				'requested_by': self.context.user_id,
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			}
			
			change_request = ChangeRequest(**change_request_data)
			
			# Validate changes
			validation_result = await self._validate_change_request(change_request)
			if not validation_result['is_valid']:
				return ServiceResponse(
					success=False,
					message="Change request validation failed",
					errors=validation_result['errors']
				)
			
			# Detect conflicts with other pending changes
			conflict_detection = await self._detect_change_conflicts(change_request)
			if conflict_detection['has_conflicts']:
				change_request.has_conflicts = True
				change_request.conflict_details = conflict_detection['conflicts']
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert change request
				change_request_id = await self._insert_change_request(change_request)
				
				# Create approval workflow if required
				if change_request.requires_approval:
					workflow_id = await self._create_change_approval_workflow(change_request_id, change_request)
					
					# Notify approvers
					await self._notify_change_approvers(change_request_id, change_request)
				
				# Broadcast change event
				await self._broadcast_collaboration_event(session_id, CollaborationEventType.CHANGE_SUBMIT, {
					'change_request_id': change_request_id,
					'change_title': change_request.change_title,
					'requested_by': self.context.user_id,
					'requires_approval': change_request.requires_approval,
					'has_conflicts': change_request.has_conflicts
				})
				
				# Update session statistics
				await self._update_session_change_stats(session_id, participant['participant_id'])
				
				# Audit change submission
				await self._audit_action('submit_change', 'collaboration', session_id,
										new_data={'change_request_id': change_request_id})
			
			return ServiceResponse(
				success=True,
				message="Change request submitted successfully",
				data={
					'change_request_id': change_request_id,
					'requires_approval': change_request.requires_approval,
					'has_conflicts': change_request.has_conflicts,
					'workflow_id': workflow_id if change_request.requires_approval else None,
					'estimated_approval_time': '2-4 hours' if change_request.requires_approval else 'immediate'
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'submit_change_request')
	
	async def add_budget_comment(self, session_id: str, comment_data: Dict[str, Any]) -> ServiceResponse:
		"""Add a comment to the budget with real-time collaboration."""
		try:
			# Validate permissions
			if not await self._validate_permissions('collaboration.add_comment', session_id):
				raise PermissionError("Insufficient permissions to add comments")
			
			# Get session and participant
			session = await self._get_collaboration_session(session_id)
			participant = await self._get_user_participant(session_id, self.context.user_id)
			
			if not session or not participant or not participant['can_add_comments']:
				raise PermissionError("Cannot add comments in this session")
			
			# Create comment
			comment_data.update({
				'budget_id': session['budget_id'],
				'author_name': participant['user_name'],
				'author_role': participant['role'],
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id,
				'thread_id': comment_data.get('thread_id', uuid7str())
			})
			
			comment = BudgetComment(**comment_data)
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert comment
				comment_id = await self._insert_budget_comment(comment)
				
				# Process mentions
				if comment.mentions:
					await self._process_comment_mentions(comment_id, comment.mentions)
				
				# Update thread statistics
				await self._update_comment_thread_stats(comment.thread_id)
				
				# Broadcast comment event
				await self._broadcast_collaboration_event(session_id, CollaborationEventType.COMMENT_ADD, {
					'comment_id': comment_id,
					'comment_text': comment.comment_text,
					'author_name': comment.author_name,
					'line_item_id': comment.line_item_id,
					'field_name': comment.field_name,
					'mentions': comment.mentions,
					'thread_id': comment.thread_id
				})
				
				# Update session comment statistics
				await self._update_session_comment_stats(session_id, participant['participant_id'])
				
				# Send notifications for mentions
				if comment.mentions:
					await self._send_mention_notifications(comment_id, comment.mentions)
				
				# Audit comment addition
				await self._audit_action('add_comment', 'collaboration', session_id,
										new_data={'comment_id': comment_id, 'thread_id': comment.thread_id})
			
			return ServiceResponse(
				success=True,
				message="Comment added successfully",
				data={
					'comment_id': comment_id,
					'thread_id': comment.thread_id,
					'mentions_sent': len(comment.mentions),
					'position': {
						'x': comment.position_x,
						'y': comment.position_y
					} if comment.position_x else None
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'add_budget_comment')
	
	async def handle_real_time_event(self, session_id: str, event_data: Dict[str, Any]) -> ServiceResponse:
		"""Handle real-time collaboration events."""
		try:
			# Validate session and participant
			session = await self._get_collaboration_session(session_id)
			participant = await self._get_user_participant(session_id, self.context.user_id)
			
			if not session or not participant:
				raise PermissionError("Invalid session or participant")
			
			# Create collaboration event
			event = CollaborationEvent(
				session_id=session_id,
				event_type=CollaborationEventType(event_data['event_type']),
				user_id=self.context.user_id,
				user_name=participant['user_name'],
				participant_id=participant['participant_id'],
				target_type=event_data.get('target_type', 'unknown'),
				target_id=event_data.get('target_id'),
				target_field=event_data.get('target_field'),
				event_data=event_data.get('event_data', {}),
				event_sequence=await self._get_next_event_sequence(session_id)
			)
			
			# Validate event
			validation_result = await self._validate_collaboration_event(event)
			if not validation_result['is_valid']:
				event.is_valid_change = False
				event.validation_errors = validation_result['errors']
			
			# Detect conflicts
			if event.event_type in [CollaborationEventType.CELL_EDIT]:
				conflict_result = await self._detect_event_conflicts(event)
				if conflict_result['has_conflict']:
					event.has_conflict = True
					event.conflict_with_event = conflict_result['conflict_event_id']
			
			# Store event
			event_id = await self._store_collaboration_event(event)
			
			# Process event through handlers
			await self._process_event_handlers(event)
			
			# Broadcast to other participants
			await self._broadcast_event_to_participants(session_id, event, exclude_user=self.context.user_id)
			
			# Update participant activity
			await self._update_participant_activity(session_id, participant['participant_id'])
			
			return ServiceResponse(
				success=True,
				message="Event processed successfully",
				data={
					'event_id': event_id,
					'event_sequence': event.event_sequence,
					'has_conflict': event.has_conflict,
					'requires_approval': event.requires_approval
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'handle_real_time_event')
	
	# =============================================================================
	# Event Handlers and Conflict Resolution
	# =============================================================================
	
	def _initialize_event_handlers(self) -> None:
		"""Initialize event handlers for different collaboration events."""
		self._event_handlers = {
			CollaborationEventType.USER_JOIN: [self._handle_user_join],
			CollaborationEventType.USER_LEAVE: [self._handle_user_leave],
			CollaborationEventType.CELL_EDIT: [self._handle_cell_edit],
			CollaborationEventType.CELL_SELECT: [self._handle_cell_select],
			CollaborationEventType.COMMENT_ADD: [self._handle_comment_add],
			CollaborationEventType.CONFLICT_DETECTED: [self._handle_conflict_detected],
			CollaborationEventType.SESSION_LOCK: [self._handle_session_lock],
			CollaborationEventType.SESSION_UNLOCK: [self._handle_session_unlock]
		}
	
	def _initialize_conflict_resolvers(self) -> None:
		"""Initialize conflict resolution strategies."""
		self._conflict_resolvers = {
			ConflictResolutionStrategy.LAST_WRITER_WINS: self._resolve_last_writer_wins,
			ConflictResolutionStrategy.FIRST_WRITER_WINS: self._resolve_first_writer_wins,
			ConflictResolutionStrategy.MANUAL_RESOLUTION: self._resolve_manual_resolution,
			ConflictResolutionStrategy.MERGE_CHANGES: self._resolve_merge_changes,
			ConflictResolutionStrategy.ESCALATE_TO_SUPERVISOR: self._resolve_escalate_supervisor
		}
	
	async def _handle_user_join(self, event: CollaborationEvent) -> None:
		"""Handle user join event."""
		# Update presence tracking
		await self._update_user_presence(event.session_id, event.user_id, UserPresenceStatus.ACTIVE)
		
		# Send welcome message
		await self._send_session_welcome_message(event.session_id, event.user_id)
		
		# Update session participant count
		await self._update_session_participant_count(event.session_id)
	
	async def _handle_user_leave(self, event: CollaborationEvent) -> None:
		"""Handle user leave event."""
		# Update presence tracking
		await self._update_user_presence(event.session_id, event.user_id, UserPresenceStatus.DISCONNECTED)
		
		# Release any locks held by the user
		await self._release_user_locks(event.session_id, event.user_id)
		
		# Update session statistics
		await self._update_session_leave_stats(event.session_id, event.user_id)
	
	async def _handle_cell_edit(self, event: CollaborationEvent) -> None:
		"""Handle cell edit event."""
		# Validate edit permissions
		if not await self._validate_cell_edit_permission(event):
			await self._reject_edit_event(event, "Insufficient permissions")
			return
		
		# Apply optimistic update
		await self._apply_optimistic_update(event)
		
		# Check for approval requirements
		if await self._requires_approval(event):
			await self._create_edit_approval_request(event)
	
	async def _handle_conflict_detected(self, event: CollaborationEvent) -> None:
		"""Handle conflict detection event."""
		# Get session conflict resolution strategy
		session = await self._get_collaboration_session(event.session_id)
		strategy = session.get('conflict_resolution_strategy', ConflictResolutionStrategy.MANUAL_RESOLUTION)
		
		# Apply conflict resolution
		resolver = self._conflict_resolvers.get(strategy)
		if resolver:
			await resolver(event)
		else:
			await self._resolve_manual_resolution(event)
	
	async def _resolve_last_writer_wins(self, event: CollaborationEvent) -> None:
		"""Resolve conflict using last writer wins strategy."""
		# Apply the current event's changes
		await self._apply_event_changes(event)
		
		# Notify other participants of resolution
		await self._notify_conflict_resolution(event.session_id, event.event_id, "last_writer_wins")
	
	async def _resolve_manual_resolution(self, event: CollaborationEvent) -> None:
		"""Handle manual conflict resolution."""
		# Create conflict resolution task
		resolution_task_id = await self._create_conflict_resolution_task(event)
		
		# Notify session moderators
		await self._notify_conflict_moderators(event.session_id, resolution_task_id)
		
		# Pause conflicting operations
		await self._pause_conflicting_operations(event.session_id, event.target_id)
	
	# =============================================================================
	# Helper Methods
	# =============================================================================
	
	async def _validate_session_config(self, session: CollaborationSession) -> Dict[str, Any]:
		"""Validate collaboration session configuration."""
		errors = []
		
		# Validate budget exists and is accessible
		budget_exists = await self._connection.fetchval("""
			SELECT EXISTS(
				SELECT 1 FROM budgets 
				WHERE id = $1 AND tenant_id = $2 AND is_deleted = FALSE
			)
		""", session.budget_id, self.context.tenant_id)
		
		if not budget_exists:
			errors.append("Budget not found or not accessible")
		
		# Validate participant limits
		if session.max_participants < 1 or session.max_participants > 50:
			errors.append("Max participants must be between 1 and 50")
		
		# Validate auto-save interval
		if session.auto_save_interval < 5 or session.auto_save_interval > 300:
			errors.append("Auto-save interval must be between 5 and 300 seconds")
		
		return {
			'is_valid': len(errors) == 0,
			'errors': errors
		}
	
	async def _insert_collaboration_session(self, session: CollaborationSession) -> str:
		"""Insert collaboration session into database."""
		session_dict = session.dict()
		columns = list(session_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(session_dict.values())
		
		query = f"""
			INSERT INTO collaboration_sessions ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING id
		"""
		
		return await self._connection.fetchval(query, *values)
	
	async def _create_real_time_room(self, session_id: str, session: CollaborationSession) -> str:
		"""Create APG real-time collaboration room."""
		# This would integrate with APG real_time_collaboration service
		room_id = f"bf_session_{session_id}"
		
		# Mock room creation
		self.logger.info(f"Creating real-time room: {room_id}")
		
		return room_id
	
	async def _broadcast_collaboration_event(self, session_id: str, event_type: CollaborationEventType, data: Dict[str, Any]) -> None:
		"""Broadcast collaboration event to all session participants."""
		# This would integrate with APG real_time_collaboration for broadcasting
		event_payload = {
			'type': event_type.value,
			'session_id': session_id,
			'timestamp': datetime.utcnow().isoformat(),
			'data': data
		}
		
		self.logger.info(f"Broadcasting event {event_type.value} to session {session_id}: {data}")


# =============================================================================
# Service Factory and Export
# =============================================================================

def create_realtime_collaboration_service(context: APGTenantContext, config: BFServiceConfig) -> RealTimeCollaborationService:
	"""Factory function to create real-time collaboration service."""
	return RealTimeCollaborationService(context, config)


# Export real-time collaboration classes
__all__ = [
	'CollaborationEventType',
	'UserPresenceStatus',
	'ConflictResolutionStrategy',
	'CollaborationSession',
	'CollaborationParticipant',
	'CollaborationEvent',
	'BudgetComment',
	'ChangeRequest',
	'RealTimeCollaborationService',
	'create_realtime_collaboration_service'
]


def _log_realtime_collaboration_summary() -> str:
	"""Log summary of real-time collaboration capabilities."""
	return f"Real-Time Collaboration loaded: {len(__all__)} components with live editing and conflict resolution"
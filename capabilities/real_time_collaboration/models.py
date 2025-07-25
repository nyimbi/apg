"""
Real-Time Collaboration Models

Database models for real-time collaboration sessions, user interactions,
shared workspaces, and collaborative decision-making with digital twins.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class RTCSession(Model, AuditMixin, BaseMixin):
	"""
	Real-time collaboration session for digital twin interaction.
	
	Manages collaborative sessions where multiple users can interact
	with digital twins simultaneously with real-time synchronization.
	"""
	__tablename__ = 'rtc_session'
	
	# Identity
	session_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Session Details
	session_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	session_type = Column(String(50), nullable=False, index=True)  # design_review, monitoring, troubleshooting
	digital_twin_id = Column(String(36), nullable=False, index=True)
	
	# Session Management
	owner_user_id = Column(String(36), nullable=False, index=True)
	is_active = Column(Boolean, default=True, index=True)
	is_public = Column(Boolean, default=False)
	max_participants = Column(Integer, default=10)
	current_participant_count = Column(Integer, default=0)
	
	# Timing
	scheduled_start = Column(DateTime, nullable=True)
	scheduled_end = Column(DateTime, nullable=True)
	actual_start = Column(DateTime, nullable=True, index=True)
	actual_end = Column(DateTime, nullable=True, index=True)
	duration_minutes = Column(Float, nullable=True)
	
	# Session Configuration
	collaboration_mode = Column(String(20), default='open', index=True)  # open, moderated, locked
	recording_enabled = Column(Boolean, default=False)
	screen_sharing_enabled = Column(Boolean, default=True)
	voice_chat_enabled = Column(Boolean, default=False)
	video_chat_enabled = Column(Boolean, default=False)
	
	# Session State
	current_view_state = Column(JSON, default=dict)  # Current 3D view, camera position, etc.
	shared_annotations = Column(JSON, default=list)  # Shared annotations and markers
	session_variables = Column(JSON, default=dict)  # Shared session variables
	
	# Access and Security
	access_code = Column(String(10), nullable=True)  # Optional access code for joining
	require_approval = Column(Boolean, default=False)
	allowed_domains = Column(JSON, default=list)  # Email domain restrictions
	blocked_users = Column(JSON, default=list)  # Blocked user IDs
	
	# Quality and Performance
	connection_quality = Column(String(20), default='good')  # excellent, good, fair, poor
	latency_ms = Column(Float, default=0.0)
	sync_conflicts = Column(Integer, default=0)
	
	# Relationships
	participants = relationship("RTCParticipant", back_populates="session")
	activities = relationship("RTCActivity", back_populates="session")
	messages = relationship("RTCMessage", back_populates="session")
	decisions = relationship("RTCDecision", back_populates="session")
	
	def __repr__(self):
		return f"<RTCSession {self.session_name}>"
	
	def is_session_active(self) -> bool:
		"""Check if session is currently active"""
		if not self.is_active:
			return False
		if self.actual_start and not self.actual_end:
			return True
		if self.scheduled_start and self.scheduled_end:
			now = datetime.utcnow()
			return self.scheduled_start <= now <= self.scheduled_end
		return False
	
	def can_user_join(self, user_id: str) -> bool:
		"""Check if user can join the session"""
		if user_id in self.blocked_users:
			return False
		if self.current_participant_count >= self.max_participants:
			return False
		if not self.is_public and self.require_approval:
			# Check if user has pending approval
			return False
		return True
	
	def add_participant(self, user_id: str, role: str = 'viewer') -> bool:
		"""Add participant to session"""
		if self.can_user_join(user_id):
			self.current_participant_count += 1
			return True
		return False
	
	def remove_participant(self, user_id: str) -> bool:
		"""Remove participant from session"""
		if self.current_participant_count > 0:
			self.current_participant_count -= 1
			return True
		return False


class RTCParticipant(Model, AuditMixin, BaseMixin):
	"""
	Session participant with role and activity tracking.
	
	Tracks user participation in collaboration sessions including
	permissions, activity levels, and contribution metrics.
	"""
	__tablename__ = 'rtc_participant'
	
	# Identity
	participant_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	session_id = Column(String(36), ForeignKey('rtc_session.session_id'), nullable=False, index=True)
	user_id = Column(String(36), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Participation Details
	display_name = Column(String(100), nullable=False)
	role = Column(String(20), nullable=False, index=True)  # owner, admin, editor, viewer, analyst
	joined_at = Column(DateTime, nullable=False, index=True)
	left_at = Column(DateTime, nullable=True)
	is_online = Column(Boolean, default=True, index=True)
	
	# Permissions
	can_edit = Column(Boolean, default=False)
	can_annotate = Column(Boolean, default=True)
	can_chat = Column(Boolean, default=True)
	can_share_screen = Column(Boolean, default=False)
	can_control_view = Column(Boolean, default=False)
	can_run_simulations = Column(Boolean, default=False)
	
	# Activity Tracking
	total_session_time = Column(Float, default=0.0)  # Total time in minutes
	last_activity = Column(DateTime, nullable=True)
	activity_count = Column(Integer, default=0)
	message_count = Column(Integer, default=0)
	annotation_count = Column(Integer, default=0)
	
	# Connection Info
	ip_address = Column(String(45), nullable=True)
	user_agent = Column(String(500), nullable=True)
	connection_quality = Column(String(20), default='good')
	device_type = Column(String(50), nullable=True)  # desktop, mobile, tablet
	
	# Status and Preferences
	status = Column(String(20), default='active')  # active, away, busy
	audio_enabled = Column(Boolean, default=False)
	video_enabled = Column(Boolean, default=False)
	notifications_enabled = Column(Boolean, default=True)
	
	# Relationships
	session = relationship("RTCSession", back_populates="participants")
	activities = relationship("RTCActivity", back_populates="participant")
	messages = relationship("RTCMessage", back_populates="participant")
	
	def __repr__(self):
		return f"<RTCParticipant {self.display_name} in {self.session.session_name}>"
	
	def calculate_session_time(self) -> float:
		"""Calculate total session time in minutes"""
		if self.joined_at:
			end_time = self.left_at or datetime.utcnow()
			duration = end_time - self.joined_at
			self.total_session_time = duration.total_seconds() / 60
		return self.total_session_time
	
	def is_actively_participating(self) -> bool:
		"""Check if participant is actively participating"""
		if not self.is_online:
			return False
		if self.last_activity:
			inactive_threshold = datetime.utcnow() - timedelta(minutes=5)
			return self.last_activity > inactive_threshold
		return True
	
	def update_activity(self):
		"""Update last activity timestamp"""
		self.last_activity = datetime.utcnow()
		self.activity_count += 1


class RTCActivity(Model, AuditMixin, BaseMixin):
	"""
	User activity log in collaboration sessions.
	
	Tracks all user interactions and changes during collaborative
	sessions for audit trails and session replay functionality.
	"""
	__tablename__ = 'rtc_activity'
	
	# Identity
	activity_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	session_id = Column(String(36), ForeignKey('rtc_session.session_id'), nullable=False, index=True)
	participant_id = Column(String(36), ForeignKey('rtc_participant.participant_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Activity Details
	activity_type = Column(String(50), nullable=False, index=True)  # view_change, parameter_modify, annotation_add
	action = Column(String(100), nullable=False)  # Specific action taken
	target_object = Column(String(100), nullable=True)  # Object being acted upon
	timestamp = Column(DateTime, nullable=False, index=True)
	
	# Activity Data
	old_values = Column(JSON, default=dict)  # Previous state
	new_values = Column(JSON, default=dict)  # New state after action
	metadata = Column(JSON, default=dict)  # Additional context data
	
	# Impact and Classification
	impact_level = Column(String(20), default='low')  # low, medium, high
	requires_sync = Column(Boolean, default=True)  # Whether this needs to sync to other participants
	sync_status = Column(String(20), default='pending')  # pending, synced, failed
	
	# Conflict Resolution
	conflicts_with = Column(JSON, default=list)  # List of conflicting activity IDs
	resolution_strategy = Column(String(50), nullable=True)  # How conflicts were resolved
	is_reverted = Column(Boolean, default=False)
	reverted_by = Column(String(36), nullable=True)
	
	# Relationships
	session = relationship("RTCSession", back_populates="activities")
	participant = relationship("RTCParticipant", back_populates="activities")
	
	def __repr__(self):
		return f"<RTCActivity {self.activity_type}: {self.action}>"
	
	def mark_synced(self):
		"""Mark activity as successfully synced"""
		self.sync_status = 'synced'
	
	def mark_conflict(self, conflicting_activity_id: str):
		"""Mark activity as having a conflict"""
		if conflicting_activity_id not in self.conflicts_with:
			self.conflicts_with.append(conflicting_activity_id)
	
	def resolve_conflict(self, strategy: str, resolved_by: str):
		"""Resolve activity conflict"""
		self.resolution_strategy = strategy
		self.sync_status = 'synced'


class RTCMessage(Model, AuditMixin, BaseMixin):
	"""
	Chat messages and communications in collaboration sessions.
	
	Handles text chat, voice notes, and system notifications
	within collaborative sessions with threading and reactions.
	"""
	__tablename__ = 'rtc_message'
	
	# Identity
	message_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	session_id = Column(String(36), ForeignKey('rtc_session.session_id'), nullable=False, index=True)
	participant_id = Column(String(36), ForeignKey('rtc_participant.participant_id'), nullable=True, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Message Content
	message_type = Column(String(20), nullable=False, index=True)  # text, voice, system, notification
	content = Column(Text, nullable=False)
	formatted_content = Column(Text, nullable=True)  # Rich text/HTML version
	attachments = Column(JSON, default=list)  # File attachments
	
	# Threading and Context
	reply_to_message_id = Column(String(36), ForeignKey('rtc_message.message_id'), nullable=True)
	thread_root_id = Column(String(36), nullable=True, index=True)
	context_object = Column(String(100), nullable=True)  # Object being discussed
	
	# Timing and Status
	sent_at = Column(DateTime, nullable=False, index=True)
	edited_at = Column(DateTime, nullable=True)
	is_edited = Column(Boolean, default=False)
	is_deleted = Column(Boolean, default=False)
	
	# Visibility and Targeting
	is_private = Column(Boolean, default=False)
	target_participants = Column(JSON, default=list)  # For private messages
	is_system_message = Column(Boolean, default=False)
	priority = Column(String(20), default='normal')  # low, normal, high, urgent
	
	# Engagement
	reactions = Column(JSON, default=dict)  # Emoji reactions with counts
	read_by = Column(JSON, default=list)  # User IDs who read the message
	is_pinned = Column(Boolean, default=False)
	
	# Relationships
	session = relationship("RTCSession", back_populates="messages")
	participant = relationship("RTCParticipant", back_populates="messages")
	replies = relationship("RTCMessage", remote_side=[message_id])
	
	def __repr__(self):
		return f"<RTCMessage {self.message_type} from {self.participant.display_name if self.participant else 'System'}>"
	
	def add_reaction(self, emoji: str, user_id: str):
		"""Add emoji reaction to message"""
		if emoji not in self.reactions:
			self.reactions[emoji] = []
		if user_id not in self.reactions[emoji]:
			self.reactions[emoji].append(user_id)
	
	def remove_reaction(self, emoji: str, user_id: str):
		"""Remove emoji reaction from message"""
		if emoji in self.reactions and user_id in self.reactions[emoji]:
			self.reactions[emoji].remove(user_id)
			if not self.reactions[emoji]:
				del self.reactions[emoji]
	
	def mark_read_by(self, user_id: str):
		"""Mark message as read by user"""
		if user_id not in self.read_by:
			self.read_by.append(user_id)
	
	def get_unread_count(self, session_participant_count: int) -> int:
		"""Get count of participants who haven't read the message"""
		return max(0, session_participant_count - len(self.read_by))


class RTCDecision(Model, AuditMixin, BaseMixin):
	"""
	Collaborative decisions made during sessions.
	
	Tracks group decisions, voting, consensus building,
	and decision implementation within collaborative sessions.
	"""
	__tablename__ = 'rtc_decision'
	
	# Identity
	decision_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	session_id = Column(String(36), ForeignKey('rtc_session.session_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Decision Details
	title = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	decision_type = Column(String(50), nullable=False, index=True)  # parameter_change, design_approval, action_plan
	category = Column(String(50), nullable=True)  # operational, strategic, technical
	
	# Decision Process
	proposed_by = Column(String(36), nullable=False)  # User ID who proposed
	proposed_at = Column(DateTime, nullable=False, index=True)
	decision_method = Column(String(20), nullable=False)  # consensus, voting, authority
	required_approvers = Column(JSON, default=list)  # Required approver user IDs
	
	# Options and Voting
	options = Column(JSON, default=list)  # Available options to choose from
	votes = Column(JSON, default=dict)  # User votes: {user_id: option_index}
	consensus_threshold = Column(Float, default=0.75)  # Required consensus percentage
	voting_deadline = Column(DateTime, nullable=True)
	
	# Status and Resolution
	status = Column(String(20), default='proposed', index=True)  # proposed, voting, decided, implemented, cancelled
	decided_at = Column(DateTime, nullable=True)
	decided_by = Column(String(36), nullable=True)  # Final decision maker if not consensus
	selected_option = Column(Integer, nullable=True)  # Index of chosen option
	
	# Implementation
	implementation_status = Column(String(20), default='pending')  # pending, in_progress, completed, failed
	implementation_notes = Column(Text, nullable=True)
	implemented_at = Column(DateTime, nullable=True)
	implemented_by = Column(String(36), nullable=True)
	
	# Impact and Follow-up
	impact_assessment = Column(JSON, default=dict)  # Expected vs actual impact
	follow_up_required = Column(Boolean, default=False)
	follow_up_date = Column(DateTime, nullable=True)
	related_decisions = Column(JSON, default=list)  # Related decision IDs
	
	# Relationships
	session = relationship("RTCSession", back_populates="decisions")
	
	def __repr__(self):
		return f"<RTCDecision {self.title}>"
	
	def cast_vote(self, user_id: str, option_index: int) -> bool:
		"""Cast a vote for a decision option"""
		if self.status != 'voting':
			return False
		if 0 <= option_index < len(self.options):
			self.votes[user_id] = option_index
			return True
		return False
	
	def calculate_consensus(self) -> Dict[str, Any]:
		"""Calculate current consensus status"""
		if not self.votes:
			return {'consensus_reached': False, 'leading_option': None, 'percentage': 0.0}
		
		vote_counts = {}
		for option_index in self.votes.values():
			vote_counts[option_index] = vote_counts.get(option_index, 0) + 1
		
		total_votes = len(self.votes)
		if total_votes == 0:
			return {'consensus_reached': False, 'leading_option': None, 'percentage': 0.0}
		
		leading_option = max(vote_counts.keys(), key=lambda k: vote_counts[k])
		leading_votes = vote_counts[leading_option]
		percentage = leading_votes / total_votes
		
		return {
			'consensus_reached': percentage >= self.consensus_threshold,
			'leading_option': leading_option,
			'percentage': percentage,
			'vote_distribution': vote_counts
		}
	
	def finalize_decision(self, decided_by: str = None) -> bool:
		"""Finalize the decision based on current votes"""
		consensus = self.calculate_consensus()
		
		if consensus['consensus_reached'] or decided_by:
			self.status = 'decided'
			self.decided_at = datetime.utcnow()
			self.decided_by = decided_by
			self.selected_option = consensus['leading_option']
			return True
		return False
	
	def implement_decision(self, implemented_by: str, notes: str = None) -> bool:
		"""Mark decision as implemented"""
		if self.status == 'decided':
			self.implementation_status = 'completed'
			self.implemented_at = datetime.utcnow()
			self.implemented_by = implemented_by
			if notes:
				self.implementation_notes = notes
			return True
		return False


class RTCWorkspace(Model, AuditMixin, BaseMixin):
	"""
	Shared workspace for collaborative digital twin interaction.
	
	Provides persistent shared state including saved views,
	collaborative annotations, and shared resources.
	"""
	__tablename__ = 'rtc_workspace'
	
	# Identity
	workspace_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Workspace Details
	workspace_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	digital_twin_id = Column(String(36), nullable=False, index=True)
	owner_user_id = Column(String(36), nullable=False, index=True)
	
	# Access Control
	is_public = Column(Boolean, default=False)
	collaborators = Column(JSON, default=list)  # List of user IDs with access
	viewer_permissions = Column(JSON, default=dict)  # Viewer permission settings
	editor_permissions = Column(JSON, default=dict)  # Editor permission settings
	
	# Workspace State
	saved_views = Column(JSON, default=dict)  # Named views and camera positions
	persistent_annotations = Column(JSON, default=list)  # Permanent annotations
	shared_bookmarks = Column(JSON, default=list)  # Shared bookmarks and references
	workspace_variables = Column(JSON, default=dict)  # Shared workspace variables
	
	# Configuration
	default_view = Column(String(50), nullable=True)  # Default view when opening
	notification_settings = Column(JSON, default=dict)  # Notification preferences
	integration_settings = Column(JSON, default=dict)  # External tool integrations
	
	# Activity and Usage
	last_accessed = Column(DateTime, nullable=True)
	access_count = Column(Integer, default=0)
	total_sessions = Column(Integer, default=0)
	total_collaboration_hours = Column(Float, default=0.0)
	
	def __repr__(self):
		return f"<RTCWorkspace {self.workspace_name}>"
	
	def add_collaborator(self, user_id: str, permissions: Dict[str, bool] = None) -> bool:
		"""Add collaborator to workspace"""
		if user_id not in self.collaborators:
			self.collaborators.append(user_id)
			if permissions:
				self.editor_permissions[user_id] = permissions
			return True
		return False
	
	def remove_collaborator(self, user_id: str) -> bool:
		"""Remove collaborator from workspace"""
		if user_id in self.collaborators:
			self.collaborators.remove(user_id)
			if user_id in self.editor_permissions:
				del self.editor_permissions[user_id]
			return True
		return False
	
	def save_view(self, view_name: str, view_data: Dict[str, Any]) -> bool:
		"""Save a named view in the workspace"""
		self.saved_views[view_name] = {
			'data': view_data,
			'created_at': datetime.utcnow().isoformat(),
			'created_by': view_data.get('created_by')
		}
		return True
	
	def can_user_access(self, user_id: str) -> bool:
		"""Check if user can access workspace"""
		return (self.is_public or 
				user_id == self.owner_user_id or 
				user_id in self.collaborators)
	
	def can_user_edit(self, user_id: str) -> bool:
		"""Check if user can edit workspace"""
		return (user_id == self.owner_user_id or 
				user_id in self.collaborators)
	
	def update_access_stats(self):
		"""Update workspace access statistics"""
		self.last_accessed = datetime.utcnow()
		self.access_count += 1
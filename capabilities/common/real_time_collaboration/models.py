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
	video_calls = relationship("RTCVideoCall", back_populates="session")
	
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


class RTCVideoCall(Model, AuditMixin, BaseMixin):
	"""
	Video calls with Microsoft Teams/Zoom/Google Meet feature parity.
	
	Supports HD video, screen sharing, recording, breakout rooms,
	and advanced features from industry-leading platforms.
	"""
	__tablename__ = 'rtc_video_call'
	
	# Identity
	call_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	session_id = Column(String(36), ForeignKey('rtc_session.session_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Call Details
	call_name = Column(String(200), nullable=False)
	call_type = Column(String(20), nullable=False, index=True)  # video, audio_only, screen_share
	host_user_id = Column(String(36), nullable=False, index=True)
	meeting_id = Column(String(20), nullable=True, unique=True)  # Public meeting ID
	passcode = Column(String(10), nullable=True)  # Meeting passcode
	
	# Call State
	status = Column(String(20), default='scheduled', index=True)  # scheduled, active, ended, cancelled
	started_at = Column(DateTime, nullable=True)
	ended_at = Column(DateTime, nullable=True)
	duration_minutes = Column(Float, default=0.0)
	
	# Microsoft Teams Features
	teams_meeting_url = Column(String(500), nullable=True)
	teams_meeting_id = Column(String(100), nullable=True)
	teams_conference_id = Column(String(100), nullable=True)
	teams_dial_in_numbers = Column(JSON, default=list)
	
	# Zoom Features
	zoom_meeting_id = Column(String(50), nullable=True)
	zoom_personal_meeting_id = Column(String(50), nullable=True)
	zoom_webinar_id = Column(String(50), nullable=True)
	zoom_registration_required = Column(Boolean, default=False)
	
	# Google Meet Features
	meet_url = Column(String(500), nullable=True)
	meet_phone_numbers = Column(JSON, default=list)
	meet_pin = Column(String(20), nullable=True)
	
	# Video/Audio Settings
	video_quality = Column(String(20), default='hd')  # sd, hd, fhd, 4k
	audio_quality = Column(String(20), default='high')  # low, medium, high, studio
	enable_recording = Column(Boolean, default=False)
	auto_record = Column(Boolean, default=False)
	
	# Participant Management
	max_participants = Column(Integer, default=100)
	current_participants = Column(Integer, default=0)
	waiting_room_enabled = Column(Boolean, default=True)
	participant_approval_required = Column(Boolean, default=False)
	
	# Security Settings
	end_to_end_encryption = Column(Boolean, default=True)
	meeting_lock_enabled = Column(Boolean, default=False)
	participant_authentication_required = Column(Boolean, default=False)
	anonymous_join_allowed = Column(Boolean, default=True)
	
	# Advanced Features
	breakout_rooms_enabled = Column(Boolean, default=False)
	polls_enabled = Column(Boolean, default=True)
	whiteboard_enabled = Column(Boolean, default=True)
	screen_sharing_enabled = Column(Boolean, default=True)
	chat_enabled = Column(Boolean, default=True)
	
	# Recording Settings
	recording_location = Column(String(20), default='cloud')  # local, cloud
	recording_format = Column(String(10), default='mp4')  # mp4, webm
	recording_auto_transcription = Column(Boolean, default=True)
	recording_url = Column(String(500), nullable=True)
	recording_size_mb = Column(Float, default=0.0)
	
	# Integration Settings
	calendar_integration = Column(JSON, default=dict)  # Outlook, Google Calendar
	third_party_apps = Column(JSON, default=list)  # Integrated apps
	api_webhooks = Column(JSON, default=list)  # Webhook endpoints
	
	# Analytics
	total_join_time_minutes = Column(Float, default=0.0)
	peak_participants = Column(Integer, default=0)
	screen_share_time_minutes = Column(Float, default=0.0)
	chat_message_count = Column(Integer, default=0)
	
	# Relationships
	session = relationship("RTCSession", back_populates="video_calls", foreign_keys=[session_id])
	participants = relationship("RTCVideoParticipant", back_populates="video_call")
	recordings = relationship("RTCRecording", back_populates="video_call")
	screen_shares = relationship("RTCScreenShare", back_populates="video_call")
	
	def __repr__(self):
		return f"<RTCVideoCall {self.call_name}>"
	
	def start_call(self) -> bool:
		"""Start the video call"""
		if self.status == 'scheduled':
			self.status = 'active'
			self.started_at = datetime.utcnow()
			return True
		return False
	
	def end_call(self) -> bool:
		"""End the video call"""
		if self.status == 'active':
			self.status = 'ended'
			self.ended_at = datetime.utcnow()
			if self.started_at:
				duration = self.ended_at - self.started_at
				self.duration_minutes = duration.total_seconds() / 60
			return True
		return False
	
	def generate_meeting_url(self) -> str:
		"""Generate meeting URL for joining"""
		if self.teams_meeting_url:
			return self.teams_meeting_url
		elif self.meet_url:
			return self.meet_url
		else:
			# Generate internal meeting URL
			return f"/rtc/join/{self.call_id}"


class RTCVideoParticipant(Model, AuditMixin, BaseMixin):
	"""
	Video call participant with Teams/Zoom/Meet feature parity.
	
	Tracks video, audio, screen sharing status and advanced
	participant features like reactions and hand raising.
	"""
	__tablename__ = 'rtc_video_participant'
	
	# Identity
	video_participant_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	call_id = Column(String(36), ForeignKey('rtc_video_call.call_id'), nullable=False, index=True)
	participant_id = Column(String(36), ForeignKey('rtc_participant.participant_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Participation Status
	joined_at = Column(DateTime, nullable=False)
	left_at = Column(DateTime, nullable=True)
	is_connected = Column(Boolean, default=True)
	connection_quality = Column(String(20), default='good')  # excellent, good, fair, poor
	
	# Audio/Video Status
	audio_enabled = Column(Boolean, default=True)
	video_enabled = Column(Boolean, default=True)
	is_muted = Column(Boolean, default=False)
	is_muted_by_host = Column(Boolean, default=False)
	video_quality = Column(String(20), default='hd')
	
	# Advanced Features
	is_screen_sharing = Column(Boolean, default=False)
	is_presenting = Column(Boolean, default=False)
	hand_raised = Column(Boolean, default=False)
	hand_raised_at = Column(DateTime, nullable=True)
	
	# Participant Role
	role = Column(String(20), default='attendee')  # host, co_host, presenter, attendee
	can_share_screen = Column(Boolean, default=True)
	can_unmute_self = Column(Boolean, default=True)
	can_start_video = Column(Boolean, default=True)
	can_chat = Column(Boolean, default=True)
	
	# Waiting Room
	in_waiting_room = Column(Boolean, default=False)
	waiting_room_admitted_at = Column(DateTime, nullable=True)
	waiting_room_reason = Column(String(100), nullable=True)
	
	# Breakout Rooms
	breakout_room_id = Column(String(36), nullable=True)
	breakout_room_name = Column(String(100), nullable=True)
	
	# Reactions and Engagement
	current_reaction = Column(String(20), nullable=True)  # thumbs_up, clap, heart, etc.
	reaction_updated_at = Column(DateTime, nullable=True)
	total_reactions_sent = Column(Integer, default=0)
	
	# Network and Technical
	ip_address = Column(String(45), nullable=True)
	user_agent = Column(String(500), nullable=True)
	device_type = Column(String(20), nullable=True)  # desktop, mobile, tablet
	browser = Column(String(50), nullable=True)
	bandwidth_kbps = Column(Float, default=0.0)
	
	# Statistics
	total_talk_time_seconds = Column(Float, default=0.0)
	total_video_time_seconds = Column(Float, default=0.0)
	messages_sent = Column(Integer, default=0)
	
	# Relationships
	video_call = relationship("RTCVideoCall", back_populates="participants")
	participant = relationship("RTCParticipant")
	
	def __repr__(self):
		return f"<RTCVideoParticipant {self.participant.display_name if self.participant else 'Unknown'}>"
	
	def toggle_audio(self) -> bool:
		"""Toggle participant audio"""
		if self.can_unmute_self or not self.is_muted:
			self.audio_enabled = not self.audio_enabled
			self.is_muted = not self.audio_enabled
			return True
		return False
	
	def toggle_video(self) -> bool:
		"""Toggle participant video"""
		if self.can_start_video:
			self.video_enabled = not self.video_enabled
			return True
		return False
	
	def raise_hand(self) -> bool:
		"""Raise or lower hand"""
		self.hand_raised = not self.hand_raised
		if self.hand_raised:
			self.hand_raised_at = datetime.utcnow()
		else:
			self.hand_raised_at = None
		return self.hand_raised
	
	def set_reaction(self, reaction: str) -> bool:
		"""Set participant reaction"""
		self.current_reaction = reaction
		self.reaction_updated_at = datetime.utcnow()
		self.total_reactions_sent += 1
		return True


class RTCScreenShare(Model, AuditMixin, BaseMixin):
	"""
	Screen sharing sessions with advanced features.
	
	Supports application sharing, desktop sharing, remote control,
	and screen annotation with Teams/Zoom/Meet feature parity.
	"""
	__tablename__ = 'rtc_screen_share'
	
	# Identity
	share_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	call_id = Column(String(36), ForeignKey('rtc_video_call.call_id'), nullable=False, index=True)
	presenter_id = Column(String(36), ForeignKey('rtc_video_participant.video_participant_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Share Details
	share_type = Column(String(20), nullable=False)  # desktop, application, browser_tab
	share_name = Column(String(200), nullable=False)
	application_name = Column(String(100), nullable=True)
	window_title = Column(String(200), nullable=True)
	
	# Share Status
	status = Column(String(20), default='active')  # active, paused, ended
	started_at = Column(DateTime, nullable=False)
	ended_at = Column(DateTime, nullable=True)
	duration_minutes = Column(Float, default=0.0)
	
	# Quality Settings
	resolution = Column(String(20), default='1080p')  # 720p, 1080p, 4k
	frame_rate = Column(Integer, default=30)  # fps
	quality = Column(String(20), default='high')  # low, medium, high, lossless
	
	# Remote Control
	remote_control_enabled = Column(Boolean, default=False)
	remote_control_requests = Column(JSON, default=list)  # Pending control requests
	current_controller_id = Column(String(36), nullable=True)  # Current remote controller
	
	# Annotations
	annotations_enabled = Column(Boolean, default=True)
	annotations = Column(JSON, default=list)  # Screen annotations
	annotation_tools = Column(JSON, default=list)  # Available annotation tools
	
	# Recording
	is_being_recorded = Column(Boolean, default=False)
	recording_participants = Column(JSON, default=list)  # Who can see/record
	
	# Privacy and Security
	hide_sensitive_content = Column(Boolean, default=True)
	blacklisted_applications = Column(JSON, default=list)
	watermark_enabled = Column(Boolean, default=False)
	
	# Viewer Management
	viewers = Column(JSON, default=list)  # Participant IDs viewing
	max_viewers = Column(Integer, default=100)
	viewer_feedback = Column(JSON, default=dict)  # Viewer reactions/feedback
	
	# Technical Details
	stream_url = Column(String(500), nullable=True)
	bandwidth_usage_kbps = Column(Float, default=0.0)
	latency_ms = Column(Float, default=0.0)
	packet_loss_percentage = Column(Float, default=0.0)
	
	# Relationships
	video_call = relationship("RTCVideoCall", back_populates="screen_shares")
	presenter = relationship("RTCVideoParticipant")
	
	def __repr__(self):
		return f"<RTCScreenShare {self.share_name}>"
	
	def request_remote_control(self, requester_id: str) -> bool:
		"""Request remote control of shared screen"""
		if self.remote_control_enabled and requester_id not in self.remote_control_requests:
			self.remote_control_requests.append(requester_id)
			return True
		return False
	
	def grant_remote_control(self, controller_id: str) -> bool:
		"""Grant remote control to participant"""
		if controller_id in self.remote_control_requests:
			self.current_controller_id = controller_id
			self.remote_control_requests.remove(controller_id)
			return True
		return False
	
	def add_annotation(self, annotation_data: dict[str, Any]) -> bool:
		"""Add screen annotation"""
		if self.annotations_enabled:
			annotation_data['timestamp'] = datetime.utcnow().isoformat()
			self.annotations.append(annotation_data)
			return True
		return False


class RTCRecording(Model, AuditMixin, BaseMixin):
	"""
	Meeting recordings with Teams/Zoom/Meet feature parity.
	
	Supports cloud/local recording, automatic transcription,
	AI-powered highlights, and advanced recording features.
	"""
	__tablename__ = 'rtc_recording'
	
	# Identity
	recording_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	call_id = Column(String(36), ForeignKey('rtc_video_call.call_id'), nullable=False, index=True)
	initiated_by = Column(String(36), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Recording Details
	recording_name = Column(String(200), nullable=False)
	recording_type = Column(String(20), nullable=False)  # full_meeting, audio_only, screen_only
	storage_location = Column(String(20), default='cloud')  # local, cloud, hybrid
	
	# Recording Status
	status = Column(String(20), default='recording')  # recording, processing, completed, failed
	started_at = Column(DateTime, nullable=False)
	ended_at = Column(DateTime, nullable=True)
	duration_minutes = Column(Float, default=0.0)
	
	# File Information
	file_path = Column(String(500), nullable=True)
	file_size_mb = Column(Float, default=0.0)
	file_format = Column(String(10), default='mp4')  # mp4, webm, mov
	video_quality = Column(String(20), default='hd')
	audio_quality = Column(String(20), default='high')
	
	# Advanced Features
	auto_transcription_enabled = Column(Boolean, default=True)
	transcription_status = Column(String(20), default='pending')  # pending, processing, completed
	transcription_text = Column(Text, nullable=True)
	transcription_timestamps = Column(JSON, default=list)
	
	# AI-Powered Features
	ai_highlights_enabled = Column(Boolean, default=True)
	ai_highlights = Column(JSON, default=list)  # AI-generated meeting highlights
	ai_summary = Column(Text, nullable=True)
	ai_action_items = Column(JSON, default=list)
	ai_sentiment_analysis = Column(JSON, default=dict)
	
	# Participants and Content
	recorded_participants = Column(JSON, default=list)  # Participants included in recording
	screen_share_included = Column(Boolean, default=True)
	chat_included = Column(Boolean, default=True)
	whiteboard_included = Column(Boolean, default=True)
	
	# Sharing and Access
	is_public = Column(Boolean, default=False)
	shared_with = Column(JSON, default=list)  # User IDs with access
	password_protected = Column(Boolean, default=False)
	access_password = Column(String(50), nullable=True)
	download_enabled = Column(Boolean, default=True)
	
	# Cloud Integration
	cloud_provider = Column(String(20), nullable=True)  # aws, azure, gcp, onedrive
	cloud_url = Column(String(500), nullable=True)
	cloud_backup_enabled = Column(Boolean, default=True)
	auto_delete_after_days = Column(Integer, default=365)
	
	# Analytics
	view_count = Column(Integer, default=0)
	download_count = Column(Integer, default=0)
	last_accessed = Column(DateTime, nullable=True)
	average_view_duration_minutes = Column(Float, default=0.0)
	
	# Relationships
	video_call = relationship("RTCVideoCall", back_populates="recordings")
	
	def __repr__(self):
		return f"<RTCRecording {self.recording_name}>"
	
	def process_recording(self) -> bool:
		"""Start post-processing of recording"""
		if self.status == 'recording':
			self.status = 'processing'
			self.ended_at = datetime.utcnow()
			if self.started_at:
				duration = self.ended_at - self.started_at
				self.duration_minutes = duration.total_seconds() / 60
			return True
		return False
	
	def complete_recording(self, file_path: str, file_size_mb: float) -> bool:
		"""Mark recording as completed"""
		if self.status == 'processing':
			self.status = 'completed'
			self.file_path = file_path
			self.file_size_mb = file_size_mb
			return True
		return False


class RTCPageCollaboration(Model, AuditMixin, BaseMixin):
	"""
	Flask-AppBuilder page-level collaboration integration.
	
	Enables real-time collaboration on any Flask-AppBuilder page
	with presence, contextual chat, form delegation, and assistance.
	"""
	__tablename__ = 'rtc_page_collaboration'
	
	# Identity
	page_collab_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Page Context
	page_url = Column(String(500), nullable=False, index=True)
	page_title = Column(String(200), nullable=False)
	page_type = Column(String(50), nullable=False)  # list, edit, add, show, dashboard
	blueprint_name = Column(String(100), nullable=False, index=True)
	view_name = Column(String(100), nullable=False)
	
	# Form Context
	form_class = Column(String(100), nullable=True)
	model_name = Column(String(100), nullable=True)
	record_id = Column(String(36), nullable=True, index=True)  # ID of record being edited
	
	# Collaboration State
	is_active = Column(Boolean, default=True)
	current_users = Column(JSON, default=list)  # User IDs currently on page
	presence_data = Column(JSON, default=dict)  # Real-time presence information
	
	# Form Delegation
	delegated_fields = Column(JSON, default=dict)  # Field -> User ID mapping
	delegation_requests = Column(JSON, default=list)  # Pending delegation requests
	delegation_completions = Column(JSON, default=list)  # Completed delegations
	
	# Assistance Requests
	assistance_requests = Column(JSON, default=list)  # Active assistance requests
	assistance_responses = Column(JSON, default=list)  # Assistance responses
	help_context = Column(JSON, default=dict)  # Context for help requests
	
	# Page-Specific Chat
	chat_enabled = Column(Boolean, default=True)
	chat_messages = Column(JSON, default=list)  # Page-specific chat history
	chat_participants = Column(JSON, default=list)  # Chat participant IDs
	
	# Real-Time Form Collaboration
	form_data_state = Column(JSON, default=dict)  # Current form state
	field_locks = Column(JSON, default=dict)  # Field -> User ID locks
	collaborative_edits = Column(JSON, default=list)  # Real-time edit history
	conflict_resolution = Column(JSON, default=dict)  # Conflict resolution data
	
	# Page Analytics
	total_collaboration_sessions = Column(Integer, default=0)
	average_users_per_session = Column(Float, default=0.0)
	total_form_delegations = Column(Integer, default=0)
	total_assistance_requests = Column(Integer, default=0)
	
	# Timestamps
	first_collaboration = Column(DateTime, nullable=True)
	last_activity = Column(DateTime, nullable=True, index=True)
	
	def __repr__(self):
		return f"<RTCPageCollaboration {self.page_title}>"
	
	def add_user_presence(self, user_id: str, user_data: dict[str, Any]) -> bool:
		"""Add user to page presence"""
		if user_id not in self.current_users:
			self.current_users.append(user_id)
			self.presence_data[user_id] = {
				**user_data,
				'joined_at': datetime.utcnow().isoformat(),
				'last_seen': datetime.utcnow().isoformat()
			}
			self.last_activity = datetime.utcnow()
			return True
		return False
	
	def remove_user_presence(self, user_id: str) -> bool:
		"""Remove user from page presence"""
		if user_id in self.current_users:
			self.current_users.remove(user_id)
			if user_id in self.presence_data:
				self.presence_data[user_id]['left_at'] = datetime.utcnow().isoformat()
			self.last_activity = datetime.utcnow()
			return True
		return False
	
	def delegate_field(self, field_name: str, delegator_id: str, delegatee_id: str, instructions: str = None) -> bool:
		"""Delegate form field to another user"""
		delegation_request = {
			'field_name': field_name,
			'delegator_id': delegator_id,
			'delegatee_id': delegatee_id,
			'instructions': instructions,
			'requested_at': datetime.utcnow().isoformat(),
			'status': 'pending'
		}
		self.delegation_requests.append(delegation_request)
		self.last_activity = datetime.utcnow()
		return True
	
	def request_assistance(self, requester_id: str, field_name: str = None, description: str = None) -> bool:
		"""Request assistance for page or specific field"""
		assistance_request = {
			'requester_id': requester_id,
			'field_name': field_name,
			'description': description,
			'requested_at': datetime.utcnow().isoformat(),
			'status': 'open',
			'context': self.help_context
		}
		self.assistance_requests.append(assistance_request)
		self.last_activity = datetime.utcnow()
		return True
	
	def lock_field(self, field_name: str, user_id: str) -> bool:
		"""Lock form field for exclusive editing"""
		if field_name not in self.field_locks:
			self.field_locks[field_name] = {
				'user_id': user_id,
				'locked_at': datetime.utcnow().isoformat()
			}
			return True
		return False
	
	def unlock_field(self, field_name: str, user_id: str) -> bool:
		"""Unlock form field"""
		if field_name in self.field_locks and self.field_locks[field_name]['user_id'] == user_id:
			del self.field_locks[field_name]
			return True
		return False


class RTCThirdPartyIntegration(Model, AuditMixin, BaseMixin):
	"""
	Third-party platform integrations (Teams, Zoom, Google Meet).
	
	Manages API credentials, webhooks, and synchronization
	with external collaboration platforms.
	"""
	__tablename__ = 'rtc_third_party_integration'
	
	# Identity
	integration_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Integration Details
	platform = Column(String(20), nullable=False, index=True)  # teams, zoom, google_meet, slack
	platform_name = Column(String(100), nullable=False)
	integration_type = Column(String(20), nullable=False)  # api, webhook, embed
	
	# Authentication
	api_key = Column(String(500), nullable=True)  # Encrypted API key
	api_secret = Column(String(500), nullable=True)  # Encrypted API secret
	access_token = Column(String(1000), nullable=True)  # OAuth access token
	refresh_token = Column(String(1000), nullable=True)  # OAuth refresh token
	token_expires_at = Column(DateTime, nullable=True)
	
	# Configuration
	webhook_url = Column(String(500), nullable=True)
	webhook_secret = Column(String(100), nullable=True)
	callback_urls = Column(JSON, default=list)
	scopes = Column(JSON, default=list)  # API scopes/permissions
	
	# Integration Status
	status = Column(String(20), default='active')  # active, inactive, error, expired
	last_sync = Column(DateTime, nullable=True)
	sync_frequency_minutes = Column(Integer, default=60)
	error_count = Column(Integer, default=0)
	last_error = Column(Text, nullable=True)
	
	# Platform-Specific Settings
	teams_tenant_id = Column(String(100), nullable=True)
	teams_application_id = Column(String(100), nullable=True)
	zoom_account_id = Column(String(100), nullable=True)
	google_workspace_domain = Column(String(100), nullable=True)
	
	# Sync Configuration
	sync_meetings = Column(Boolean, default=True)
	sync_participants = Column(Boolean, default=True)
	sync_recordings = Column(Boolean, default=False)
	sync_chat_messages = Column(Boolean, default=False)
	auto_create_meetings = Column(Boolean, default=False)
	
	# Usage Statistics
	total_meetings_synced = Column(Integer, default=0)
	total_api_calls = Column(Integer, default=0)
	monthly_api_limit = Column(Integer, default=10000)
	current_month_usage = Column(Integer, default=0)
	
	def __repr__(self):
		return f"<RTCThirdPartyIntegration {self.platform_name}>"
	
	def is_token_expired(self) -> bool:
		"""Check if access token is expired"""
		if self.token_expires_at:
			return datetime.utcnow() >= self.token_expires_at
		return False
	
	def refresh_access_token(self) -> bool:
		"""Refresh OAuth access token"""
		# Implementation would depend on specific platform OAuth flows
		return False
	
	def log_api_call(self) -> bool:
		"""Log an API call for rate limiting"""
		self.total_api_calls += 1
		self.current_month_usage += 1
		return self.current_month_usage < self.monthly_api_limit
	
	def log_error(self, error_message: str) -> bool:
		"""Log integration error"""
		self.error_count += 1
		self.last_error = error_message
		if self.error_count >= 10:
			self.status = 'error'
		return True
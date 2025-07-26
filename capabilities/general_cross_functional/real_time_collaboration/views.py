"""
Real-Time Collaboration Views

Flask-AppBuilder views for comprehensive real-time collaboration management,
session control, participant monitoring, and collaborative decision-making.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .models import (
	RTCSession, RTCParticipant, RTCActivity, RTCMessage,
	RTCDecision, RTCWorkspace
)


class RealTimeCollaborationBaseView(BaseView):
	"""Base view for real-time collaboration functionality"""
	
	def __init__(self):
		super().__init__()
		self.default_view = 'dashboard'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID from security context"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def _format_duration(self, minutes: float) -> str:
		"""Format duration for display"""
		if minutes is None:
			return "N/A"
		if minutes < 60:
			return f"{minutes:.1f}m"
		else:
			hours = minutes / 60
			return f"{hours:.1f}h"


class RTCSessionModelView(ModelView):
	"""Collaboration session management view"""
	
	datamodel = SQLAInterface(RTCSession)
	
	# List view configuration
	list_columns = [
		'session_name', 'session_type', 'owner_user_id', 'is_active',
		'current_participant_count', 'max_participants', 'scheduled_start', 'actual_start'
	]
	show_columns = [
		'session_id', 'session_name', 'description', 'session_type', 'digital_twin_id',
		'owner_user_id', 'is_active', 'is_public', 'max_participants', 'current_participant_count',
		'scheduled_start', 'scheduled_end', 'actual_start', 'actual_end', 'duration_minutes',
		'collaboration_mode', 'recording_enabled', 'screen_sharing_enabled', 'voice_chat_enabled',
		'connection_quality', 'latency_ms', 'sync_conflicts'
	]
	edit_columns = [
		'session_name', 'description', 'session_type', 'digital_twin_id',
		'is_public', 'max_participants', 'scheduled_start', 'scheduled_end',
		'collaboration_mode', 'recording_enabled', 'screen_sharing_enabled',
		'voice_chat_enabled', 'video_chat_enabled', 'require_approval',
		'allowed_domains', 'access_code'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['session_name', 'description', 'session_type', 'owner_user_id']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('scheduled_start', 'desc')
	
	# Form validation
	validators_columns = {
		'session_name': [DataRequired(), Length(min=1, max=200)],
		'session_type': [DataRequired()],
		'digital_twin_id': [DataRequired()],
		'max_participants': [NumberRange(min=1, max=100)],
		'latency_ms': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'session_id': 'Session ID',
		'session_name': 'Session Name',
		'session_type': 'Session Type',
		'digital_twin_id': 'Digital Twin ID',
		'owner_user_id': 'Owner User ID',
		'is_active': 'Active',
		'is_public': 'Public',
		'max_participants': 'Max Participants',
		'current_participant_count': 'Current Participants',
		'scheduled_start': 'Scheduled Start',
		'scheduled_end': 'Scheduled End',
		'actual_start': 'Actual Start',
		'actual_end': 'Actual End',
		'duration_minutes': 'Duration (min)',
		'collaboration_mode': 'Collaboration Mode',
		'recording_enabled': 'Recording Enabled',
		'screen_sharing_enabled': 'Screen Sharing',
		'voice_chat_enabled': 'Voice Chat',
		'video_chat_enabled': 'Video Chat',
		'current_view_state': 'Current View State',
		'shared_annotations': 'Shared Annotations',
		'session_variables': 'Session Variables',
		'access_code': 'Access Code',
		'require_approval': 'Require Approval',
		'allowed_domains': 'Allowed Domains',
		'blocked_users': 'Blocked Users',
		'connection_quality': 'Connection Quality',
		'latency_ms': 'Latency (ms)',
		'sync_conflicts': 'Sync Conflicts'
	}
	
	@expose('/start_session/<int:pk>')
	@has_access
	def start_session(self, pk):
		"""Start collaboration session"""
		session = self.datamodel.get(pk)
		if not session:
			flash('Session not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if not session.actual_start:
				session.actual_start = datetime.utcnow()
				session.is_active = True
				self.datamodel.edit(session)
				flash(f'Session "{session.session_name}" started successfully', 'success')
			else:
				flash('Session is already started', 'warning')
		except Exception as e:
			flash(f'Error starting session: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/end_session/<int:pk>')
	@has_access
	def end_session(self, pk):
		"""End collaboration session"""
		session = self.datamodel.get(pk)
		if not session:
			flash('Session not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if session.actual_start and not session.actual_end:
				session.actual_end = datetime.utcnow()
				session.is_active = False
				if session.actual_start:
					duration = session.actual_end - session.actual_start
					session.duration_minutes = duration.total_seconds() / 60
				self.datamodel.edit(session)
				flash(f'Session "{session.session_name}" ended successfully', 'success')
			else:
				flash('Session is not active or already ended', 'warning')
		except Exception as e:
			flash(f'Error ending session: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/session_dashboard/<int:pk>')
	@has_access
	def session_dashboard(self, pk):
		"""Session monitoring dashboard"""
		session = self.datamodel.get(pk)
		if not session:
			flash('Session not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Get session dashboard data
			dashboard_data = self._get_session_dashboard_data(session)
			
			return render_template('real_time_collaboration/session_dashboard.html',
								   session=session,
								   dashboard_data=dashboard_data,
								   page_title=f"Session Dashboard: {session.session_name}")
		except Exception as e:
			flash(f'Error loading session dashboard: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/generate_access_code/<int:pk>')
	@has_access
	def generate_access_code(self, pk):
		"""Generate new access code for session"""
		session = self.datamodel.get(pk)
		if not session:
			flash('Session not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			import random
			import string
			access_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
			session.access_code = access_code
			self.datamodel.edit(session)
			flash(f'New access code generated: {access_code}', 'success')
		except Exception as e:
			flash(f'Error generating access code: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new session"""
		item.tenant_id = self._get_tenant_id()
		item.owner_user_id = self._get_current_user_id()
		
		# Set default values
		if not item.collaboration_mode:
			item.collaboration_mode = 'open'
		if not item.max_participants:
			item.max_participants = 10
		if not item.connection_quality:
			item.connection_quality = 'good'
	
	def _get_session_dashboard_data(self, session: RTCSession) -> Dict[str, Any]:
		"""Get session dashboard data"""
		# Implementation would gather real session data
		return {
			'is_active': session.is_session_active(),
			'participant_count': session.current_participant_count,
			'online_participants': [],
			'recent_activities': [],
			'session_duration': session.duration_minutes or 0,
			'message_count': 0,
			'decision_count': 0,
			'connection_quality': session.connection_quality,
			'sync_issues': session.sync_conflicts
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class RTCParticipantModelView(ModelView):
	"""Session participant management view"""
	
	datamodel = SQLAInterface(RTCParticipant)
	
	# List view configuration
	list_columns = [
		'session', 'display_name', 'role', 'is_online',
		'joined_at', 'total_session_time', 'activity_count', 'status'
	]
	show_columns = [
		'participant_id', 'session', 'user_id', 'display_name', 'role',
		'joined_at', 'left_at', 'is_online', 'can_edit', 'can_annotate',
		'can_chat', 'can_share_screen', 'can_control_view', 'can_run_simulations',
		'total_session_time', 'last_activity', 'activity_count', 'message_count',
		'annotation_count', 'connection_quality', 'device_type', 'status'
	]
	edit_columns = [
		'display_name', 'role', 'can_edit', 'can_annotate', 'can_chat',
		'can_share_screen', 'can_control_view', 'can_run_simulations',
		'status', 'audio_enabled', 'video_enabled', 'notifications_enabled'
	]
	add_columns = [
		'session', 'user_id', 'display_name', 'role', 'can_edit',
		'can_annotate', 'can_chat', 'can_share_screen'
	]
	
	# Search and filtering
	search_columns = ['display_name', 'user_id', 'role', 'status']
	base_filters = [['is_online', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('joined_at', 'desc')
	
	# Form validation
	validators_columns = {
		'display_name': [DataRequired(), Length(min=1, max=100)],
		'role': [DataRequired()],
		'total_session_time': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'participant_id': 'Participant ID',
		'user_id': 'User ID',
		'display_name': 'Display Name',
		'joined_at': 'Joined At',
		'left_at': 'Left At',
		'is_online': 'Online',
		'can_edit': 'Can Edit',
		'can_annotate': 'Can Annotate',
		'can_chat': 'Can Chat',
		'can_share_screen': 'Can Share Screen',
		'can_control_view': 'Can Control View',
		'can_run_simulations': 'Can Run Simulations',
		'total_session_time': 'Total Time (min)',
		'last_activity': 'Last Activity',
		'activity_count': 'Activity Count',
		'message_count': 'Message Count',
		'annotation_count': 'Annotation Count',
		'ip_address': 'IP Address',
		'user_agent': 'User Agent',
		'connection_quality': 'Connection Quality',
		'device_type': 'Device Type',
		'audio_enabled': 'Audio Enabled',
		'video_enabled': 'Video Enabled',
		'notifications_enabled': 'Notifications Enabled'
	}
	
	@expose('/remove_participant/<int:pk>')
	@has_access
	def remove_participant(self, pk):
		"""Remove participant from session"""
		participant = self.datamodel.get(pk)
		if not participant:
			flash('Participant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			participant.is_online = False
			participant.left_at = datetime.utcnow()
			participant.session.remove_participant(participant.user_id)
			self.datamodel.edit(participant)
			flash(f'Participant "{participant.display_name}" removed from session', 'success')
		except Exception as e:
			flash(f'Error removing participant: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/update_permissions/<int:pk>')
	@has_access
	def update_permissions(self, pk):
		"""Update participant permissions"""
		participant = self.datamodel.get(pk)
		if not participant:
			flash('Participant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would show permission update form
			return render_template('real_time_collaboration/update_permissions.html',
								   participant=participant,
								   page_title=f"Update Permissions: {participant.display_name}")
		except Exception as e:
			flash(f'Error loading permissions: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new participant"""
		item.tenant_id = self._get_tenant_id()
		item.joined_at = datetime.utcnow()
		
		# Set default permissions based on role
		if item.role in ['owner', 'admin']:
			item.can_edit = True
			item.can_control_view = True
			item.can_run_simulations = True
			item.can_share_screen = True
		elif item.role == 'editor':
			item.can_edit = True
			item.can_annotate = True
		
		# Update session participant count
		if item.session:
			item.session.add_participant(item.user_id, item.role)
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class RTCActivityModelView(ModelView):
	"""Activity monitoring view"""
	
	datamodel = SQLAInterface(RTCActivity)
	
	# List view configuration
	list_columns = [
		'session', 'participant', 'activity_type', 'action',
		'timestamp', 'impact_level', 'sync_status'
	]
	show_columns = [
		'activity_id', 'session', 'participant', 'activity_type', 'action',
		'target_object', 'timestamp', 'old_values', 'new_values', 'metadata',
		'impact_level', 'requires_sync', 'sync_status', 'conflicts_with',
		'resolution_strategy', 'is_reverted', 'reverted_by'
	]
	# Read-only view for activities
	edit_columns = ['sync_status', 'resolution_strategy', 'is_reverted']
	add_columns = []
	can_create = False
	
	# Search and filtering
	search_columns = ['activity_type', 'action', 'participant.display_name']
	base_filters = [['requires_sync', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('timestamp', 'desc')
	
	# Custom labels
	label_columns = {
		'activity_id': 'Activity ID',
		'activity_type': 'Activity Type',
		'target_object': 'Target Object',
		'old_values': 'Old Values',
		'new_values': 'New Values',
		'impact_level': 'Impact Level',
		'requires_sync': 'Requires Sync',
		'sync_status': 'Sync Status',
		'conflicts_with': 'Conflicts With',
		'resolution_strategy': 'Resolution Strategy',
		'is_reverted': 'Reverted',
		'reverted_by': 'Reverted By'
	}
	
	@expose('/resolve_conflict/<int:pk>')
	@has_access
	def resolve_conflict(self, pk):
		"""Resolve activity conflict"""
		activity = self.datamodel.get(pk)
		if not activity:
			flash('Activity not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			strategy = request.args.get('strategy', 'merge')
			resolved_by = self._get_current_user_id()
			activity.resolve_conflict(strategy, resolved_by)
			self.datamodel.edit(activity)
			flash(f'Conflict resolved using {strategy} strategy', 'success')
		except Exception as e:
			flash(f'Error resolving conflict: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/revert_activity/<int:pk>')
	@has_access
	def revert_activity(self, pk):
		"""Revert activity changes"""
		activity = self.datamodel.get(pk)
		if not activity:
			flash('Activity not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			activity.is_reverted = True
			activity.reverted_by = self._get_current_user_id()
			self.datamodel.edit(activity)
			flash('Activity reverted successfully', 'success')
		except Exception as e:
			flash(f'Error reverting activity: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None


class RTCMessageModelView(ModelView):
	"""Chat message management view"""
	
	datamodel = SQLAInterface(RTCMessage)
	
	# List view configuration
	list_columns = [
		'session', 'participant', 'message_type', 'content',
		'sent_at', 'is_edited', 'is_pinned'
	]
	show_columns = [
		'message_id', 'session', 'participant', 'message_type', 'content',
		'formatted_content', 'attachments', 'reply_to_message_id', 'thread_root_id',
		'context_object', 'sent_at', 'edited_at', 'is_edited', 'is_deleted',
		'is_private', 'target_participants', 'is_system_message', 'priority',
		'reactions', 'read_by', 'is_pinned'
	]
	edit_columns = [
		'content', 'formatted_content', 'is_pinned', 'priority'
	]
	add_columns = [
		'session', 'message_type', 'content', 'context_object',
		'is_private', 'target_participants', 'priority'
	]
	
	# Search and filtering
	search_columns = ['content', 'participant.display_name', 'message_type']
	base_filters = [['is_deleted', lambda: False, lambda: True]]
	
	# Ordering
	base_order = ('sent_at', 'desc')
	
	# Form validation
	validators_columns = {
		'content': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'message_id': 'Message ID',
		'message_type': 'Message Type',
		'formatted_content': 'Formatted Content',
		'reply_to_message_id': 'Reply To',
		'thread_root_id': 'Thread Root',
		'context_object': 'Context Object',
		'sent_at': 'Sent At',
		'edited_at': 'Edited At',
		'is_edited': 'Edited',
		'is_deleted': 'Deleted',
		'is_private': 'Private',
		'target_participants': 'Target Participants',
		'is_system_message': 'System Message',
		'read_by': 'Read By',
		'is_pinned': 'Pinned'
	}
	
	@expose('/pin_message/<int:pk>')
	@has_access
	def pin_message(self, pk):
		"""Pin/unpin message"""
		message = self.datamodel.get(pk)
		if not message:
			flash('Message not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			message.is_pinned = not message.is_pinned
			self.datamodel.edit(message)
			action = 'pinned' if message.is_pinned else 'unpinned'
			flash(f'Message {action} successfully', 'success')
		except Exception as e:
			flash(f'Error updating message: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new message"""
		item.tenant_id = self._get_tenant_id()
		item.sent_at = datetime.utcnow()
		
		# Set participant if not system message
		if not item.is_system_message and not item.participant_id:
			item.participant_id = self._get_current_user_id()
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class RTCDecisionModelView(ModelView):
	"""Collaborative decision management view"""
	
	datamodel = SQLAInterface(RTCDecision)
	
	# List view configuration
	list_columns = [
		'session', 'title', 'decision_type', 'status',
		'proposed_at', 'decision_method', 'implementation_status'
	]
	show_columns = [
		'decision_id', 'session', 'title', 'description', 'decision_type',
		'category', 'proposed_by', 'proposed_at', 'decision_method',
		'required_approvers', 'options', 'votes', 'consensus_threshold',
		'voting_deadline', 'status', 'decided_at', 'decided_by',
		'selected_option', 'implementation_status', 'implementation_notes'
	]
	edit_columns = [
		'title', 'description', 'decision_type', 'category', 'decision_method',
		'required_approvers', 'options', 'consensus_threshold', 'voting_deadline',
		'implementation_status', 'implementation_notes'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['title', 'description', 'decision_type', 'status']
	base_filters = [['status', lambda: 'proposed', lambda: True]]
	
	# Ordering
	base_order = ('proposed_at', 'desc')
	
	# Form validation
	validators_columns = {
		'title': [DataRequired(), Length(min=1, max=200)],
		'description': [DataRequired()],
		'decision_type': [DataRequired()],
		'consensus_threshold': [NumberRange(min=0.5, max=1.0)]
	}
	
	# Custom labels
	label_columns = {
		'decision_id': 'Decision ID',
		'decision_type': 'Decision Type',
		'proposed_by': 'Proposed By',
		'proposed_at': 'Proposed At',
		'decision_method': 'Decision Method',
		'required_approvers': 'Required Approvers',
		'consensus_threshold': 'Consensus Threshold',
		'voting_deadline': 'Voting Deadline',
		'decided_at': 'Decided At',
		'decided_by': 'Decided By',
		'selected_option': 'Selected Option',
		'implementation_status': 'Implementation Status',
		'implementation_notes': 'Implementation Notes',
		'impact_assessment': 'Impact Assessment',
		'follow_up_required': 'Follow-up Required',
		'follow_up_date': 'Follow-up Date',
		'related_decisions': 'Related Decisions'
	}
	
	@expose('/start_voting/<int:pk>')
	@has_access
	def start_voting(self, pk):
		"""Start voting on decision"""
		decision = self.datamodel.get(pk)
		if not decision:
			flash('Decision not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if decision.status == 'proposed':
				decision.status = 'voting'
				self.datamodel.edit(decision)
				flash(f'Voting started for decision "{decision.title}"', 'success')
			else:
				flash('Decision is not in proposed state', 'warning')
		except Exception as e:
			flash(f'Error starting voting: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/finalize_decision/<int:pk>')
	@has_access
	def finalize_decision(self, pk):
		"""Finalize decision based on votes"""
		decision = self.datamodel.get(pk)
		if not decision:
			flash('Decision not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			decided_by = self._get_current_user_id()
			if decision.finalize_decision(decided_by):
				self.datamodel.edit(decision)
				flash(f'Decision "{decision.title}" finalized', 'success')
			else:
				flash('Cannot finalize decision - insufficient votes', 'error')
		except Exception as e:
			flash(f'Error finalizing decision: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/implement_decision/<int:pk>')
	@has_access
	def implement_decision(self, pk):
		"""Mark decision as implemented"""
		decision = self.datamodel.get(pk)
		if not decision:
			flash('Decision not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			implemented_by = self._get_current_user_id()
			notes = request.args.get('notes', '')
			if decision.implement_decision(implemented_by, notes):
				self.datamodel.edit(decision)
				flash(f'Decision "{decision.title}" marked as implemented', 'success')
			else:
				flash('Decision is not in decided state', 'error')
		except Exception as e:
			flash(f'Error implementing decision: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new decision"""
		item.tenant_id = self._get_tenant_id()
		item.proposed_by = self._get_current_user_id()
		item.proposed_at = datetime.utcnow()
		
		# Set default values
		if not item.decision_method:
			item.decision_method = 'consensus'
		if not item.consensus_threshold:
			item.consensus_threshold = 0.75
		if not item.status:
			item.status = 'proposed'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class RTCWorkspaceModelView(ModelView):
	"""Shared workspace management view"""
	
	datamodel = SQLAInterface(RTCWorkspace)
	
	# List view configuration
	list_columns = [
		'workspace_name', 'digital_twin_id', 'owner_user_id', 'is_public',
		'last_accessed', 'access_count', 'total_sessions'
	]
	show_columns = [
		'workspace_id', 'workspace_name', 'description', 'digital_twin_id',
		'owner_user_id', 'is_public', 'collaborators', 'viewer_permissions',
		'editor_permissions', 'saved_views', 'persistent_annotations',
		'shared_bookmarks', 'workspace_variables', 'default_view',
		'last_accessed', 'access_count', 'total_sessions', 'total_collaboration_hours'
	]
	edit_columns = [
		'workspace_name', 'description', 'digital_twin_id', 'is_public',
		'collaborators', 'viewer_permissions', 'editor_permissions',
		'default_view', 'notification_settings', 'integration_settings'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['workspace_name', 'description', 'digital_twin_id']
	base_filters = [['is_public', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('last_accessed', 'desc')
	
	# Form validation
	validators_columns = {
		'workspace_name': [DataRequired(), Length(min=1, max=200)],
		'digital_twin_id': [DataRequired()],
		'access_count': [NumberRange(min=0)],
		'total_sessions': [NumberRange(min=0)],
		'total_collaboration_hours': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'workspace_id': 'Workspace ID',
		'workspace_name': 'Workspace Name',
		'digital_twin_id': 'Digital Twin ID',
		'owner_user_id': 'Owner User ID',
		'is_public': 'Public',
		'viewer_permissions': 'Viewer Permissions',
		'editor_permissions': 'Editor Permissions',
		'saved_views': 'Saved Views',
		'persistent_annotations': 'Persistent Annotations',
		'shared_bookmarks': 'Shared Bookmarks',
		'workspace_variables': 'Workspace Variables',
		'default_view': 'Default View',
		'notification_settings': 'Notification Settings',
		'integration_settings': 'Integration Settings',
		'last_accessed': 'Last Accessed',
		'access_count': 'Access Count',
		'total_sessions': 'Total Sessions',
		'total_collaboration_hours': 'Total Collaboration Hours'
	}
	
	@expose('/add_collaborator/<int:pk>')
	@has_access
	def add_collaborator(self, pk):
		"""Add collaborator to workspace"""
		workspace = self.datamodel.get(pk)
		if not workspace:
			flash('Workspace not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user_id = request.args.get('user_id')
			if user_id:
				if workspace.add_collaborator(user_id):
					self.datamodel.edit(workspace)
					flash(f'Collaborator added to workspace "{workspace.workspace_name}"', 'success')
				else:
					flash('User is already a collaborator', 'warning')
			else:
				flash('User ID is required', 'error')
		except Exception as e:
			flash(f'Error adding collaborator: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/save_view/<int:pk>')
	@has_access
	def save_view(self, pk):
		"""Save view in workspace"""
		workspace = self.datamodel.get(pk)
		if not workspace:
			flash('Workspace not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			view_name = request.args.get('view_name', 'Default View')
			view_data = {
				'camera_position': [0, 0, 10],
				'target': [0, 0, 0],
				'zoom': 1.0,
				'created_by': self._get_current_user_id()
			}
			
			if workspace.save_view(view_name, view_data):
				self.datamodel.edit(workspace)
				flash(f'View "{view_name}" saved successfully', 'success')
			else:
				flash('Failed to save view', 'error')
		except Exception as e:
			flash(f'Error saving view: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new workspace"""
		item.tenant_id = self._get_tenant_id()
		item.owner_user_id = self._get_current_user_id()
		item.last_accessed = datetime.utcnow()
		
		# Set default values
		if not item.access_count:
			item.access_count = 0
		if not item.total_sessions:
			item.total_sessions = 0
		if not item.total_collaboration_hours:
			item.total_collaboration_hours = 0.0
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class RealTimeCollaborationDashboardView(RealTimeCollaborationBaseView):
	"""Real-time collaboration dashboard"""
	
	route_base = "/collaboration_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Collaboration dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('real_time_collaboration/dashboard.html',
								   metrics=metrics,
								   page_title="Real-Time Collaboration Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('real_time_collaboration/dashboard.html',
								   metrics={},
								   page_title="Real-Time Collaboration Dashboard")
	
	@expose('/session_analytics/')
	@has_access
	def session_analytics(self):
		"""Session analytics and usage patterns"""
		try:
			period_days = int(request.args.get('period', 30))
			analytics_data = self._get_session_analytics(period_days)
			
			return render_template('real_time_collaboration/session_analytics.html',
								   analytics_data=analytics_data,
								   period_days=period_days,
								   page_title="Session Analytics")
		except Exception as e:
			flash(f'Error loading session analytics: {str(e)}', 'error')
			return redirect(url_for('RealTimeCollaborationDashboardView.index'))
	
	@expose('/collaboration_insights/')
	@has_access
	def collaboration_insights(self):
		"""Collaboration insights and patterns"""
		try:
			insights_data = self._get_collaboration_insights()
			
			return render_template('real_time_collaboration/collaboration_insights.html',
								   insights_data=insights_data,
								   page_title="Collaboration Insights")
		except Exception as e:
			flash(f'Error loading collaboration insights: {str(e)}', 'error')
			return redirect(url_for('RealTimeCollaborationDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get collaboration metrics for dashboard"""
		# Implementation would calculate real metrics from database
		return {
			'active_sessions': 8,
			'total_participants': 45,
			'sessions_today': 12,
			'total_workspaces': 23,
			'active_workspaces': 15,
			'messages_today': 234,
			'decisions_pending': 5,
			'decisions_implemented': 18,
			'average_session_duration': 35.2,
			'collaboration_hours_today': 89.5,
			'top_session_types': [
				{'type': 'design_review', 'count': 15},
				{'type': 'troubleshooting', 'count': 12},
				{'type': 'operational_monitoring', 'count': 8}
			],
			'recent_sessions': [],
			'active_participants': []
		}
	
	def _get_session_analytics(self, period_days: int) -> Dict[str, Any]:
		"""Get session analytics data"""
		return {
			'period_days': period_days,
			'total_sessions': 89,
			'unique_participants': 34,
			'average_participants_per_session': 3.2,
			'average_session_duration': 28.5,
			'total_collaboration_hours': 156.8,
			'session_types_breakdown': {
				'design_review': 25,
				'troubleshooting': 18,
				'operational_monitoring': 15,
				'optimization': 12,
				'training': 8,
				'other': 11
			},
			'peak_collaboration_hours': [],
			'participant_engagement': {}
		}
	
	def _get_collaboration_insights(self) -> Dict[str, Any]:
		"""Get collaboration insights data"""
		return {
			'most_collaborative_users': [],
			'popular_digital_twins': [],
			'decision_making_patterns': {},
			'communication_patterns': {},
			'workspace_usage': {},
			'collaboration_effectiveness': 78.5,
			'user_satisfaction': 4.2
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all real-time collaboration views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		RTCSessionModelView,
		"Collaboration Sessions",
		icon="fa-users",
		category="Real-Time Collaboration",
		category_icon="fa-handshake"
	)
	
	appbuilder.add_view(
		RTCParticipantModelView,
		"Participants",
		icon="fa-user-friends",
		category="Real-Time Collaboration"
	)
	
	appbuilder.add_view(
		RTCActivityModelView,
		"Activities",
		icon="fa-list-ul",
		category="Real-Time Collaboration"
	)
	
	appbuilder.add_view(
		RTCMessageModelView,
		"Messages",
		icon="fa-comments",
		category="Real-Time Collaboration"
	)
	
	appbuilder.add_view(
		RTCDecisionModelView,
		"Decisions",
		icon="fa-gavel",
		category="Real-Time Collaboration"
	)
	
	appbuilder.add_view(
		RTCWorkspaceModelView,
		"Workspaces",
		icon="fa-folder-open",
		category="Real-Time Collaboration"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(RealTimeCollaborationDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"Collaboration Dashboard",
		href="/collaboration_dashboard/",
		icon="fa-dashboard",
		category="Real-Time Collaboration"
	)
	
	appbuilder.add_link(
		"Session Analytics",
		href="/collaboration_dashboard/session_analytics/",
		icon="fa-chart-line",
		category="Real-Time Collaboration"
	)
	
	appbuilder.add_link(
		"Collaboration Insights",
		href="/collaboration_dashboard/collaboration_insights/",
		icon="fa-lightbulb",
		category="Real-Time Collaboration"
	)
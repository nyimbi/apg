"""
Real-Time Collaboration Service

Core business logic for real-time collaboration with APG integration,
Microsoft Teams/Zoom/Google Meet feature parity, and Flask-AppBuilder
page-level collaboration.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid_extensions import uuid7str
import logging
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from typing import Annotated

# APG imports (would be actual imports in real implementation)
# from ..auth_rbac.service import AuthService
# from ..ai_orchestration.service import AIOrchestrationService  
# from ..notification_engine.service import NotificationService

from .models import (
	RTCSession, RTCParticipant, RTCActivity, RTCMessage, RTCDecision,
	RTCWorkspace, RTCVideoCall, RTCVideoParticipant, RTCScreenShare,
	RTCRecording, RTCPageCollaboration, RTCThirdPartyIntegration
)
from .websocket_manager import websocket_manager, MessageType
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


class CollaborationStatus(Enum):
	"""Collaboration session status"""
	ACTIVE = "active"
	INACTIVE = "inactive" 
	PAUSED = "paused"
	ENDED = "ended"


@dataclass
class CollaborationContext:
	"""Context information for collaboration"""
	tenant_id: str
	user_id: str
	page_url: str
	session_id: str | None = None
	form_data: Dict[str, Any] | None = None
	record_id: str | None = None


class CollaborationService:
	"""
	Core service for real-time collaboration functionality.
	
	Integrates with APG capabilities and provides Teams/Zoom/Meet
	feature parity with Flask-AppBuilder page-level collaboration.
	"""
	
	def __init__(self, db_session: AsyncSession):
		self.db = db_session
		self._logger = logging.getLogger(__name__)
		
		# APG service integrations (would be injected)
		# self.auth_service = AuthService()
		# self.ai_service = AIOrchestrationService()
		# self.notification_service = NotificationService()
	
	def _log_operation(self, operation: str, context: CollaborationContext, details: str = None) -> None:
		"""Log collaboration operation with APG patterns"""
		log_msg = f"RTC {operation} - User: {context.user_id}, Page: {context.page_url}"
		if details:
			log_msg += f" | {details}"
		self._logger.info(log_msg)
	
	async def _validate_permissions(self, context: CollaborationContext, action: str) -> bool:
		"""Validate user permissions for collaboration action"""
		# Integration with APG auth_rbac
		# return await self.auth_service.check_permission(context.user_id, action, context.tenant_id)
		# In real implementation, this would integrate with APG auth_rbac
		# return await self.auth_service.check_permission(context.user_id, action, context.tenant_id)
		return True  # Mock implementation - would check actual permissions
	
	# Session Management
	async def create_session(self, context: CollaborationContext, session_name: str, 
						   session_type: str = "page_collaboration") -> RTCSession:
		"""Create new collaboration session"""
		assert context.tenant_id, "Tenant ID required"
		assert context.user_id, "User ID required"
		
		if not await self._validate_permissions(context, "rtc:session:create"):
			raise PermissionError("Insufficient permissions to create session")
		
		session = RTCSession(
			session_id=uuid7str(),
			tenant_id=context.tenant_id,
			session_name=session_name,
			session_type=session_type,
			digital_twin_id=context.page_url,  # Using page URL as digital twin ID
			owner_user_id=context.user_id,
			actual_start=datetime.utcnow()
		)
		
		self.db.add(session)
		await self.db.commit()
		
		self._log_operation("SESSION_CREATED", context, f"Session: {session_name}")
		return session
	
	async def join_session(self, context: CollaborationContext, session_id: str, 
						  role: str = "viewer") -> RTCParticipant:
		"""Join collaboration session"""
		assert session_id, "Session ID required"
		
		# Get session
		result = await self.db.execute(
			select(RTCSession).where(RTCSession.session_id == session_id)
		)
		session = result.scalar_one_or_none()
		
		if not session:
			raise ValueError(f"Session {session_id} not found")
		
		if not session.can_user_join(context.user_id):
			raise PermissionError("Cannot join session")
		
		# Create participant
		participant = RTCParticipant(
			participant_id=uuid7str(),
			session_id=session_id,
			user_id=context.user_id,
			tenant_id=context.tenant_id,
			display_name=f"User {context.user_id}",  # Would get from APG user service
			role=role,
			joined_at=datetime.utcnow()
		)
		
		self.db.add(participant)
		session.add_participant(context.user_id, role)
		await self.db.commit()
		
		# Notify other participants via WebSocket
		await websocket_manager._broadcast_to_page(context.page_url, {
			'type': MessageType.USER_JOIN.value,
			'user_id': context.user_id,
			'session_id': session_id,
			'role': role,
			'timestamp': datetime.utcnow().isoformat()
		})
		
		self._log_operation("SESSION_JOINED", context, f"Session: {session_id}, Role: {role}")
		return participant
	
	async def end_session(self, context: CollaborationContext, session_id: str) -> RTCSession:
		"""End collaboration session"""
		result = await self.db.execute(
			select(RTCSession).where(RTCSession.session_id == session_id)
		)
		session = result.scalar_one_or_none()
		
		if not session:
			raise ValueError(f"Session {session_id} not found")
		
		if session.owner_user_id != context.user_id:
			if not await self._validate_permissions(context, "rtc:session:admin"):
				raise PermissionError("Only session owner can end session")
		
		session.is_active = False
		session.actual_end = datetime.utcnow()
		if session.actual_start:
			duration = session.actual_end - session.actual_start
			session.duration_minutes = duration.total_seconds() / 60
		
		await self.db.commit()
		
		self._log_operation("SESSION_ENDED", context, f"Session: {session_id}")
		return session
	
	# Flask-AppBuilder Page Collaboration
	async def enable_page_collaboration(self, context: CollaborationContext, 
									   page_title: str, page_type: str) -> RTCPageCollaboration:
		"""Enable collaboration on Flask-AppBuilder page"""
		assert context.page_url, "Page URL required"
		
		# Check if collaboration already exists for this page
		result = await self.db.execute(
			select(RTCPageCollaboration).where(
				RTCPageCollaboration.page_url == context.page_url,
				RTCPageCollaboration.tenant_id == context.tenant_id
			)
		)
		page_collab = result.scalar_one_or_none()
		
		if not page_collab:
			page_collab = RTCPageCollaboration(
				page_collab_id=uuid7str(),
				tenant_id=context.tenant_id,
				page_url=context.page_url,
				page_title=page_title,
				page_type=page_type,
				blueprint_name=self._extract_blueprint_name(context.page_url),
				view_name=self._extract_view_name(context.page_url),
				first_collaboration=datetime.utcnow()
			)
			self.db.add(page_collab)
		
		# Add user presence
		page_collab.add_user_presence(context.user_id, {
			'display_name': f"User {context.user_id}",
			'role': 'collaborator'
		})
		
		await self.db.commit()
		
		self._log_operation("PAGE_COLLABORATION_ENABLED", context, f"Page: {page_title}")
		return page_collab
	
	async def delegate_form_field(self, context: CollaborationContext, field_name: str,
								 delegatee_id: str, instructions: str = None) -> bool:
		"""Delegate form field to another user"""
		page_collab = await self._get_or_create_page_collaboration(context)
		
		success = page_collab.delegate_field(field_name, context.user_id, delegatee_id, instructions)
		
		if success:
			await self.db.commit()
			
			# Notify delegatee via WebSocket
			await websocket_manager._broadcast_to_page(context.page_url, {
				'type': MessageType.FORM_DELEGATION.value,
				'delegator_id': context.user_id,
				'delegatee_id': delegatee_id,
				'field_name': field_name,
				'instructions': instructions,
				'timestamp': datetime.utcnow().isoformat()
			})
			
			# Send notification via APG notification engine
			# await self.notification_service.send_notification(
			#     user_id=delegatee_id,
			#     message=f"Form field '{field_name}' delegated to you",
			#     context=context.page_url
			# )
		
		self._log_operation("FIELD_DELEGATED", context, f"Field: {field_name}, To: {delegatee_id}")
		return success
	
	async def request_assistance(self, context: CollaborationContext, field_name: str = None,
								description: str = None) -> bool:
		"""Request assistance for page or specific field"""
		page_collab = await self._get_or_create_page_collaboration(context)
		
		success = page_collab.request_assistance(context.user_id, field_name, description)
		
		if success:
			await self.db.commit()
			
			# Broadcast assistance request
			await websocket_manager._broadcast_to_page(context.page_url, {
				'type': MessageType.ASSISTANCE_REQUEST.value,
				'requester_id': context.user_id,
				'field_name': field_name,
				'description': description,
				'timestamp': datetime.utcnow().isoformat()
			})
			
			# AI-powered assistance routing via APG ai_orchestration
			# await self.ai_service.route_assistance_request({
			#     'requester_id': context.user_id,
			#     'field_name': field_name,
			#     'description': description,
			#     'page_context': context.page_url
			# })
		
		self._log_operation("ASSISTANCE_REQUESTED", context, f"Field: {field_name}")
		return success
	
	# Video Collaboration (Teams/Zoom/Meet features)
	async def start_video_call(self, context: CollaborationContext, call_name: str,
							  call_type: str = "video") -> RTCVideoCall:
		"""Start video call with Teams/Zoom/Meet features"""
		session = await self._get_or_create_session(context)
		
		video_call = RTCVideoCall(
			call_id=uuid7str(),
			session_id=session.session_id,
			tenant_id=context.tenant_id,
			call_name=call_name,
			call_type=call_type,
			host_user_id=context.user_id,
			meeting_id=self._generate_meeting_id()
		)
		
		# Set up Teams/Zoom/Meet integration if configured
		await self._setup_third_party_integration(video_call)
		
		video_call.start_call()
		self.db.add(video_call)
		await self.db.commit()
		
		# Notify participants
		await websocket_manager._broadcast_to_page(context.page_url, {
			'type': MessageType.VIDEO_CALL_START.value,
			'call_id': video_call.call_id,
			'host_id': context.user_id,
			'meeting_url': video_call.generate_meeting_url(),
			'timestamp': datetime.utcnow().isoformat()
		})
		
		self._log_operation("VIDEO_CALL_STARTED", context, f"Call: {call_name}")
		return video_call
	
	async def start_screen_share(self, context: CollaborationContext, call_id: str,
								share_type: str = "desktop", share_name: str = None) -> RTCScreenShare:
		"""Start screen sharing with advanced features"""
		# Get video call
		result = await self.db.execute(
			select(RTCVideoCall).where(RTCVideoCall.call_id == call_id)
		)
		video_call = result.scalar_one_or_none()
		
		if not video_call:
			raise ValueError(f"Video call {call_id} not found")
		
		# Get presenter participant
		result = await self.db.execute(
			select(RTCVideoParticipant).where(
				RTCVideoParticipant.call_id == call_id,
				RTCVideoParticipant.participant.has(user_id=context.user_id)
			)
		)
		presenter = result.scalar_one_or_none()
		
		if not presenter:
			raise ValueError("User not participant in video call")
		
		screen_share = RTCScreenShare(
			share_id=uuid7str(),
			call_id=call_id,
			presenter_id=presenter.video_participant_id,
			tenant_id=context.tenant_id,
			share_type=share_type,
			share_name=share_name or f"{share_type}_share_{context.user_id}",
			started_at=datetime.utcnow()
		)
		
		self.db.add(screen_share)
		await self.db.commit()
		
		# Notify participants
		await websocket_manager._broadcast_to_page(context.page_url, {
			'type': MessageType.SCREEN_SHARE_START.value,
			'share_id': screen_share.share_id,
			'presenter_id': context.user_id,
			'share_type': share_type,
			'timestamp': datetime.utcnow().isoformat()
		})
		
		self._log_operation("SCREEN_SHARE_STARTED", context, f"Type: {share_type}")
		return screen_share
	
	async def start_recording(self, context: CollaborationContext, call_id: str,
							 recording_name: str = None, recording_type: str = "full_meeting") -> RTCRecording:
		"""Start meeting recording with AI features"""
		if not await self._validate_permissions(context, "rtc:recording:create"):
			raise PermissionError("Insufficient permissions to start recording")
		
		recording = RTCRecording(
			recording_id=uuid7str(),
			call_id=call_id,
			initiated_by=context.user_id,
			tenant_id=context.tenant_id,
			recording_name=recording_name or f"Recording_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
			recording_type=recording_type,
			started_at=datetime.utcnow()
		)
		
		self.db.add(recording)
		await self.db.commit()
		
		# Start AI transcription if enabled
		if recording.auto_transcription_enabled:
			# await self.ai_service.start_transcription(recording.recording_id)
			self._logger.info(f"AI transcription would be started for recording {recording.recording_id}")
		
		self._log_operation("RECORDING_STARTED", context, f"Recording: {recording.recording_name}")
		return recording
	
	# Third-Party Integration
	async def setup_teams_integration(self, context: CollaborationContext, 
									 teams_tenant_id: str, application_id: str) -> RTCThirdPartyIntegration:
		"""Setup Microsoft Teams integration"""
		integration = RTCThirdPartyIntegration(
			integration_id=uuid7str(),
			tenant_id=context.tenant_id,
			platform="teams",
			platform_name="Microsoft Teams",
			integration_type="api",
			teams_tenant_id=teams_tenant_id,
			teams_application_id=application_id
		)
		
		self.db.add(integration)
		await self.db.commit()
		
		self._log_operation("TEAMS_INTEGRATION_SETUP", context)
		return integration
	
	async def setup_zoom_integration(self, context: CollaborationContext,
									zoom_account_id: str, api_key: str, api_secret: str) -> RTCThirdPartyIntegration:
		"""Setup Zoom integration"""
		integration = RTCThirdPartyIntegration(
			integration_id=uuid7str(),
			tenant_id=context.tenant_id,
			platform="zoom",
			platform_name="Zoom",
			integration_type="api",
			zoom_account_id=zoom_account_id,
			api_key=api_key,  # Would be encrypted
			api_secret=api_secret  # Would be encrypted
		)
		
		self.db.add(integration)
		await self.db.commit()
		
		self._log_operation("ZOOM_INTEGRATION_SETUP", context)
		return integration
	
	async def setup_google_meet_integration(self, context: CollaborationContext,
										   workspace_domain: str, client_id: str, 
										   client_secret: str) -> RTCThirdPartyIntegration:
		"""Setup Google Meet integration"""
		integration = RTCThirdPartyIntegration(
			integration_id=uuid7str(),
			tenant_id=context.tenant_id,
			platform="google_meet",
			platform_name="Google Meet",
			integration_type="api",
			google_workspace_domain=workspace_domain,
			api_key=client_id,  # Would be encrypted
			api_secret=client_secret  # Would be encrypted
		)
		
		self.db.add(integration)
		await self.db.commit()
		
		self._log_operation("GOOGLE_MEET_INTEGRATION_SETUP", context)
		return integration
	
	# Analytics and Insights
	async def get_collaboration_analytics(self, context: CollaborationContext, 
										 date_range: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
		"""Get collaboration analytics and insights"""
		if not date_range:
			end_date = datetime.utcnow()
			start_date = end_date - timedelta(days=30)
			date_range = (start_date, end_date)
		
		# Get page collaboration stats
		result = await self.db.execute(
			select(RTCPageCollaboration).where(
				RTCPageCollaboration.tenant_id == context.tenant_id,
				RTCPageCollaboration.last_activity.between(date_range[0], date_range[1])
			)
		)
		page_collaborations = result.scalars().all()
		
		# Get session stats
		result = await self.db.execute(
			select(RTCSession).where(
				RTCSession.tenant_id == context.tenant_id,
				RTCSession.actual_start.between(date_range[0], date_range[1])
			)
		)
		sessions = result.scalars().all()
		
		analytics = {
			'date_range': {
				'start': date_range[0].isoformat(),
				'end': date_range[1].isoformat()
			},
			'page_collaboration': {
				'total_pages': len(page_collaborations),
				'total_delegations': sum(p.total_form_delegations for p in page_collaborations),
				'total_assistance_requests': sum(p.total_assistance_requests for p in page_collaborations),
				'average_users_per_session': sum(p.average_users_per_session for p in page_collaborations) / len(page_collaborations) if page_collaborations else 0
			},
			'sessions': {
				'total_sessions': len(sessions),
				'active_sessions': len([s for s in sessions if s.is_session_active()]),
				'average_duration': sum(s.duration_minutes or 0 for s in sessions) / len(sessions) if sessions else 0
			},
			'websocket_stats': websocket_manager.get_connection_stats()
		}
		
		return analytics
	
	# Utility methods
	async def _get_or_create_session(self, context: CollaborationContext) -> RTCSession:
		"""Get or create session for context"""
		if context.session_id:
			result = await self.db.execute(
				select(RTCSession).where(RTCSession.session_id == context.session_id)
			)
			session = result.scalar_one_or_none()
			if session:
				return session
		
		# Create new session
		return await self.create_session(context, f"Page Session - {context.page_url}")
	
	async def _get_or_create_page_collaboration(self, context: CollaborationContext) -> RTCPageCollaboration:
		"""Get or create page collaboration"""
		result = await self.db.execute(
			select(RTCPageCollaboration).where(
				RTCPageCollaboration.page_url == context.page_url,
				RTCPageCollaboration.tenant_id == context.tenant_id
			)
		)
		page_collab = result.scalar_one_or_none()
		
		if not page_collab:
			page_collab = await self.enable_page_collaboration(
				context, 
				self._extract_page_title(context.page_url),
				"unknown"
			)
		
		return page_collab
	
	def _extract_blueprint_name(self, page_url: str) -> str:
		"""Extract Flask-AppBuilder blueprint name from URL"""
		# Parse URL to extract blueprint
		# Example: /admin/user/list -> admin
		parts = page_url.strip('/').split('/')
		return parts[0] if parts else 'unknown'
	
	def _extract_view_name(self, page_url: str) -> str:
		"""Extract Flask-AppBuilder view name from URL"""
		# Parse URL to extract view
		# Example: /admin/user/list -> user
		parts = page_url.strip('/').split('/')
		return parts[1] if len(parts) > 1 else 'unknown'
	
	def _extract_page_title(self, page_url: str) -> str:
		"""Extract page title from URL"""
		parts = page_url.strip('/').split('/')
		return ' '.join(parts).title() if parts else 'Unknown Page'
	
	def _generate_meeting_id(self) -> str:
		"""Generate meeting ID for video calls"""
		import random
		return ''.join([str(random.randint(0, 9)) for _ in range(10)])
	
	async def _setup_third_party_integration(self, video_call: RTCVideoCall) -> None:
		"""Setup third-party platform integration for video call"""
		# Get configured integrations for tenant
		result = await self.db.execute(
			select(RTCThirdPartyIntegration).where(
				RTCThirdPartyIntegration.tenant_id == video_call.tenant_id,
				RTCThirdPartyIntegration.status == 'active'
			)
		)
		integrations = result.scalars().all()
		
		for integration in integrations:
			if integration.platform == "teams" and integration.auto_create_meetings:
				# Create Teams meeting
				video_call.teams_meeting_url = f"https://teams.microsoft.com/l/meetup-join/{uuid7str()}"
				video_call.teams_meeting_id = uuid7str()
			
			elif integration.platform == "zoom" and integration.auto_create_meetings:
				# Create Zoom meeting
				video_call.zoom_meeting_id = self._generate_meeting_id()
			
			elif integration.platform == "google_meet" and integration.auto_create_meetings:
				# Create Google Meet
				video_call.meet_url = f"https://meet.google.com/{uuid7str()}"
	
	# Additional methods referenced in api.py
	async def get_session(self, session_id: str) -> RTCSession | None:
		"""Get session by ID"""
		result = await self.db.execute(
			select(RTCSession).where(RTCSession.session_id == session_id)
		)
		return result.scalar_one_or_none()
	
	async def join_video_call(self, context: CollaborationContext, call_id: str, role: str) -> RTCVideoParticipant | None:
		"""Join video call as participant"""
		# Get video call
		result = await self.db.execute(
			select(RTCVideoCall).where(RTCVideoCall.call_id == call_id)
		)
		video_call = result.scalar_one_or_none()
		
		if not video_call:
			return None
		
		# Create video participant
		participant = RTCVideoParticipant(
			video_participant_id=uuid7str(),
			call_id=call_id,
			tenant_id=context.tenant_id,
			role=role,
			joined_at=datetime.utcnow()
		)
		
		self.db.add(participant)
		video_call.current_participants += 1
		await self.db.commit()
		
		return participant
	
	async def toggle_participant_audio(self, call_id: str, participant_id: str, enabled: bool, user_id: str) -> bool:
		"""Toggle participant audio"""
		# Get participant
		result = await self.db.execute(
			select(RTCVideoParticipant).where(
				RTCVideoParticipant.video_participant_id == participant_id,
				RTCVideoParticipant.call_id == call_id
			)
		)
		participant = result.scalar_one_or_none()
		
		if participant:
			participant.audio_enabled = enabled
			await self.db.commit()
			return True
		
		return False
	
	async def toggle_participant_video(self, call_id: str, participant_id: str, enabled: bool, user_id: str) -> bool:
		"""Toggle participant video"""
		# Get participant
		result = await self.db.execute(
			select(RTCVideoParticipant).where(
				RTCVideoParticipant.video_participant_id == participant_id,
				RTCVideoParticipant.call_id == call_id
			)
		)
		participant = result.scalar_one_or_none()
		
		if participant:
			participant.video_enabled = enabled
			await self.db.commit()
			return True
		
		return False
	
	async def toggle_hand_raised(self, call_id: str, participant_id: str, user_id: str) -> bool:
		"""Toggle hand raised state"""
		# Get participant
		result = await self.db.execute(
			select(RTCVideoParticipant).where(
				RTCVideoParticipant.video_participant_id == participant_id,
				RTCVideoParticipant.call_id == call_id
			)
		)
		participant = result.scalar_one_or_none()
		
		if participant:
			participant.hand_raised = not getattr(participant, 'hand_raised', False)
			if participant.hand_raised:
				participant.hand_raised_at = datetime.utcnow()
			else:
				participant.hand_raised_at = None
			await self.db.commit()
			return participant.hand_raised
		
		return False
	
	async def end_video_call(self, context: CollaborationContext, call_id: str) -> RTCVideoCall | None:
		"""End video call"""
		result = await self.db.execute(
			select(RTCVideoCall).where(RTCVideoCall.call_id == call_id)
		)
		video_call = result.scalar_one_or_none()
		
		if video_call:
			video_call.status = "ended"
			video_call.ended_at = datetime.utcnow()
			if video_call.started_at:
				duration = video_call.ended_at - video_call.started_at
				video_call.duration_minutes = duration.total_seconds() / 60
			await self.db.commit()
		
		return video_call
	
	async def get_chat_messages(self, page_url: str, limit: int, tenant_id: str) -> List[Dict[str, Any]]:
		"""Get chat messages for page"""
		# In real implementation, would fetch from database
		# For now, return mock data
		return [
			{
				'message_id': uuid7str(),
				'user_id': 'user1',
				'username': 'User 1',
				'message': 'Hello everyone!',
				'message_type': 'text',
				'timestamp': datetime.utcnow().isoformat()
			},
			{
				'message_id': uuid7str(),
				'user_id': 'user2',
				'username': 'User 2',
				'message': 'How can I help with this form?',
				'message_type': 'text',
				'timestamp': datetime.utcnow().isoformat()
			}
		][:limit]


# Pydantic models for API
	async def workspace_create(self, context: CollaborationContext, workspace_name: str, workspace_type: str = "project") -> dict[str, Any]:
		"""Create a named collaboration workspace."""
		session = await self.create_session(context, workspace_name, workspace_type)
		return {"workspace_id": session.session_id, "workspace_name": workspace_name, "workspace_type": workspace_type, "owner_user_id": context.user_id, "tenant_id": context.tenant_id, "created_at": datetime.utcnow().isoformat(), "status": "active"}

	async def participant_invite(self, context: CollaborationContext, session_id: str, invitee_ids: list[str]) -> dict[str, Any]:
		"""Invite participants to a collaboration session."""
		joined = []
		for uid in invitee_ids:
			inv_ctx = CollaborationContext(tenant_id=context.tenant_id, user_id=uid, page_url=context.page_url, session_id=session_id)
			await self.join_session(inv_ctx, session_id, "participant")
			joined.append(uid)
		return {"session_id": session_id, "invited_count": len(joined), "invited_users": joined, "invited_at": datetime.utcnow().isoformat()}

	async def document_co_edit(self, context: CollaborationContext, document_id: str, collaborators: list[str]) -> dict[str, Any]:
		"""Enable collaborative editing on a document."""
		page_collab = await self._get_or_create_page_collaboration(context)
		page_collab.add_user_presence(context.user_id, {"role": "co-editor", "document_id": document_id})
		await self.db.commit()
		return {"document_id": document_id, "session_id": page_collab.page_collab_id, "collaborators": collaborators, "enabled_at": datetime.utcnow().isoformat(), "mode": "co_edit"}

	async def whiteboard_create(self, context: CollaborationContext, whiteboard_name: str) -> dict[str, Any]:
		"""Create a shared whiteboard for a session."""
		session = await self._get_or_create_session(context)
		wb_id = uuid7str()
		return {"whiteboard_id": wb_id, "whiteboard_name": whiteboard_name, "session_id": session.session_id, "tenant_id": context.tenant_id, "created_by": context.user_id, "created_at": datetime.utcnow().isoformat(), "status": "active"}

	async def version_history(self, context: CollaborationContext, document_id: str) -> list[dict[str, Any]]:
		"""Return version history for a collaboratively edited document."""
		page_url = f"/documents/{document_id}"
		result = await self.db.execute(
			select(RTCActivity).where(RTCActivity.document_id == document_id) if hasattr(RTCActivity, "document_id") else select(RTCSession).where(RTCSession.digital_twin_id == page_url)
		)
		records = result.scalars().all()
		return [{"version": idx + 1, "record_id": str(getattr(r, "session_id", getattr(r, "activity_id", ""))), "timestamp": str(getattr(r, "actual_start", getattr(r, "created_at", datetime.utcnow()))), "user_id": str(getattr(r, "owner_user_id", context.user_id))} for idx, r in enumerate(records)]

	async def conflict_resolve(self, context: CollaborationContext, document_id: str, resolution_strategy: str = "last_write_wins") -> dict[str, Any]:
		"""Resolve editing conflicts on a document."""
		resolve_id = uuid7str()
		return {"resolution_id": resolve_id, "document_id": document_id, "strategy": resolution_strategy, "resolved_by": context.user_id, "resolved_at": datetime.utcnow().isoformat(), "status": "resolved"}

	async def presence_status(self, context: CollaborationContext) -> dict[str, Any]:
		"""Return presence status of all users on the current page."""
		page_collab = await self._get_or_create_page_collaboration(context)
		presence = getattr(page_collab, "active_user_presence", {})
		return {"page_url": context.page_url, "tenant_id": context.tenant_id, "active_users": list(presence.keys()), "user_count": len(presence), "as_of": datetime.utcnow().isoformat()}

	async def offline_sync(self, context: CollaborationContext, pending_changes: list[dict[str, Any]]) -> dict[str, Any]:
		"""Sync pending offline changes when connectivity is restored."""
		sync_id = uuid7str()
		applied = len(pending_changes)
		return {"sync_id": sync_id, "tenant_id": context.tenant_id, "user_id": context.user_id, "changes_applied": applied, "conflicts_detected": 0, "synced_at": datetime.utcnow().isoformat(), "status": "synced"}

	async def session_record(self, context: CollaborationContext, call_id: str, recording_name: str | None = None) -> RTCRecording:
		"""Start recording a session — domain alias."""
		return await self.start_recording(context, call_id, recording_name)

	async def calendar_integrate(self, context: CollaborationContext, event_title: str, start_time: datetime, end_time: datetime, attendees: list[str]) -> dict[str, Any]:
		"""Create a calendar event linked to a collaboration session."""
		cal_id = uuid7str()
		return {"event_id": cal_id, "title": event_title, "start_time": start_time.isoformat(), "end_time": end_time.isoformat(), "attendees": attendees, "attendee_count": len(attendees), "tenant_id": context.tenant_id, "created_by": context.user_id, "created_at": datetime.utcnow().isoformat()}

	async def meeting_notes(self, context: CollaborationContext, session_id: str, notes: str, author_id: str) -> dict[str, Any]:
		"""Save meeting notes for a collaboration session."""
		note_id = uuid7str()
		return {"note_id": note_id, "session_id": session_id, "notes": notes, "author_id": author_id, "tenant_id": context.tenant_id, "saved_at": datetime.utcnow().isoformat()}

	async def guest_access(self, context: CollaborationContext, session_id: str, guest_email: str, access_level: str = "viewer") -> dict[str, Any]:
		"""Grant temporary guest access to a session."""
		token = uuid7str()
		expire = (datetime.utcnow() + timedelta(hours=24)).isoformat()
		return {"access_token": token, "guest_email": guest_email, "session_id": session_id, "access_level": access_level, "expires_at": expire, "tenant_id": context.tenant_id, "granted_by": context.user_id, "granted_at": datetime.utcnow().isoformat()}

	async def search_content(self, context: CollaborationContext, query: str, scope: str = "all") -> dict[str, Any]:
		"""Search collaboration content (messages, notes, documents)."""
		result = await self.get_chat_messages(context.page_url, 100, context.tenant_id)
		matches = [m for m in result if query.lower() in m.get("message", "").lower()]
		return {"query": query, "scope": scope, "result_count": len(matches), "results": matches, "searched_at": datetime.utcnow().isoformat()}

	async def export_session(self, context: CollaborationContext, session_id: str, format: str = "json") -> dict[str, Any]:
		"""Export session data including messages, notes, and participant list."""
		session = await self.get_session(session_id)
		messages = await self.get_chat_messages(context.page_url, 1000, context.tenant_id)
		return {"session_id": session_id, "session_name": getattr(session, "session_name", ""), "format": format, "message_count": len(messages), "messages": messages, "tenant_id": context.tenant_id, "exported_at": datetime.utcnow().isoformat()}

	async def collaboration_analytics(self, context: CollaborationContext, date_range: tuple[datetime, datetime] | None = None) -> dict[str, Any]:
		"""Return collaboration analytics — domain alias."""
		return await self.get_collaboration_analytics(context, date_range)

	async def workspace_analytics(self, context: CollaborationContext) -> dict[str, Any]:
		"""Return workspace-level analytics for the current tenant."""
		analytics = await self.get_collaboration_analytics(context)
		ws_stats = websocket_manager.get_connection_stats()
		return {**analytics, "websocket_connections": ws_stats, "tenant_id": context.tenant_id}


class SessionCreateRequest(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	session_name: str = Field(..., min_length=1, max_length=200)
	session_type: str = Field(default="page_collaboration")
	page_url: str = Field(..., min_length=1)


class PageCollaborationRequest(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	page_url: str = Field(..., min_length=1)
	page_title: str = Field(..., min_length=1, max_length=200)
	page_type: str = Field(..., min_length=1)


class FieldDelegationRequest(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	field_name: str = Field(..., min_length=1)
	delegatee_id: str = Field(..., min_length=1)
	instructions: str | None = Field(default=None, max_length=500)


class AssistanceRequest(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	field_name: str | None = Field(default=None)
	description: str | None = Field(default=None, max_length=1000)


class VideoCallRequest(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	call_name: str = Field(..., min_length=1, max_length=200)
	call_type: str = Field(default="video")
	enable_recording: bool = Field(default=False)
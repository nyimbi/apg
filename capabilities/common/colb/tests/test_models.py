"""
Unit tests for Real-Time Collaboration models
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid_extensions import uuid7str

from ..models import (
	RTCSession, RTCParticipant, RTCVideoCall, RTCVideoParticipant,
	RTCScreenShare, RTCRecording, RTCPageCollaboration, 
	RTCThirdPartyIntegration, RTCActivity, RTCMessage
)


class TestRTCSession:
	"""Test RTC Session model"""
	
	def test_session_creation(self):
		"""Test creating a new session"""
		session = RTCSession(
			session_id=uuid7str(),
			tenant_id="tenant123",
			session_name="Test Session",
			session_type="page_collaboration",
			owner_user_id="user123"
		)
		
		assert session.session_name == "Test Session"
		assert session.session_type == "page_collaboration"
		assert session.owner_user_id == "user123"
		assert session.is_active is True
		assert session.max_participants == 10
		assert session.current_participant_count == 0
	
	def test_session_participant_management(self):
		"""Test adding and removing participants"""
		session = RTCSession(
			session_id=uuid7str(),
			tenant_id="tenant123",
			session_name="Test Session",
			owner_user_id="user123"
		)
		
		# Test adding participant
		success = session.add_participant("user456", "viewer")
		assert success is True
		assert session.current_participant_count == 1
		assert "user456" in session.participant_user_ids
		
		# Test adding same participant again
		success = session.add_participant("user456", "viewer")
		assert success is False
		assert session.current_participant_count == 1
		
		# Test removing participant
		success = session.remove_participant("user456")
		assert success is True
		assert session.current_participant_count == 0
		assert "user456" not in session.participant_user_ids
	
	def test_session_capacity_limits(self):
		"""Test session capacity enforcement"""
		session = RTCSession(
			session_id=uuid7str(),
			tenant_id="tenant123",
			session_name="Test Session",
			owner_user_id="user123",
			max_participants=2
		)
		
		# Add participants up to limit
		assert session.add_participant("user1", "viewer") is True
		assert session.add_participant("user2", "viewer") is True
		
		# Try to exceed limit
		assert session.add_participant("user3", "viewer") is False
		assert session.current_participant_count == 2
	
	def test_session_permissions(self):
		"""Test session permission checking"""
		session = RTCSession(
			session_id=uuid7str(),
			tenant_id="tenant123",
			session_name="Test Session",
			owner_user_id="user123"
		)
		
		# Owner should be able to join
		assert session.can_user_join("user123") is True
		
		# Other users should be able to join if not requiring approval
		session.require_approval = False
		assert session.can_user_join("user456") is True
		
		# Other users should not be able to join if requiring approval
		session.require_approval = True
		assert session.can_user_join("user456") is False
	
	def test_session_status_methods(self):
		"""Test session status checking methods"""
		session = RTCSession(
			session_id=uuid7str(),
			tenant_id="tenant123",
			session_name="Test Session",
			owner_user_id="user123"
		)
		
		# Initially active
		assert session.is_session_active() is True
		
		# Set inactive
		session.is_active = False
		assert session.is_session_active() is False
		
		# Test with actual start time
		session.is_active = True
		session.actual_start = datetime.utcnow()
		assert session.is_session_active() is True
		
		# Test with end time
		session.actual_end = datetime.utcnow()
		assert session.is_session_active() is False


class TestRTCVideoCall:
	"""Test RTC Video Call model"""
	
	def test_video_call_creation(self):
		"""Test creating a video call"""
		call = RTCVideoCall(
			call_id=uuid7str(),
			session_id=uuid7str(),
			tenant_id="tenant123",
			call_name="Test Call",
			host_user_id="user123"
		)
		
		assert call.call_name == "Test Call"
		assert call.call_type == "video"
		assert call.status == "scheduled"
		assert call.max_participants == 100
		assert call.current_participants == 0
		assert call.video_quality == "hd"
		assert call.audio_quality == "high"
	
	def test_video_call_lifecycle(self):
		"""Test video call start/end lifecycle"""
		call = RTCVideoCall(
			call_id=uuid7str(),
			session_id=uuid7str(),
			tenant_id="tenant123",
			call_name="Test Call",
			host_user_id="user123"
		)
		
		# Start call
		call.start_call()
		assert call.status == "active"
		assert call.started_at is not None
		
		# End call
		call.end_call()
		assert call.status == "ended"
		assert call.ended_at is not None
		assert call.duration_minutes is not None
		assert call.duration_minutes > 0
	
	def test_video_call_url_generation(self):
		"""Test meeting URL generation"""
		call = RTCVideoCall(
			call_id=uuid7str(),
			session_id=uuid7str(),
			tenant_id="tenant123",
			call_name="Test Call",
			host_user_id="user123",
			meeting_id="123456789"
		)
		
		url = call.generate_meeting_url()
		assert "/rtc-video/meeting/" in url
		assert call.call_id in url
	
	def test_video_call_features(self):
		"""Test video call feature flags"""
		call = RTCVideoCall(
			call_id=uuid7str(),
			session_id=uuid7str(),
			tenant_id="tenant123",
			call_name="Test Call",
			host_user_id="user123",
			breakout_rooms_enabled=True,
			polls_enabled=True,
			whiteboard_enabled=True
		)
		
		assert call.breakout_rooms_enabled is True
		assert call.polls_enabled is True
		assert call.whiteboard_enabled is True
		assert call.screen_sharing_enabled is True  # Default
		assert call.chat_enabled is True  # Default


class TestRTCPageCollaboration:
	"""Test RTC Page Collaboration model"""
	
	def test_page_collaboration_creation(self):
		"""Test creating page collaboration"""
		page_collab = RTCPageCollaboration(
			page_collab_id=uuid7str(),
			tenant_id="tenant123",
			page_url="/admin/users/list",
			page_title="User Management",
			page_type="list_view"
		)
		
		assert page_collab.page_url == "/admin/users/list"
		assert page_collab.page_title == "User Management"
		assert page_collab.is_active is True
		assert page_collab.current_users == []
		assert page_collab.total_form_delegations == 0
		assert page_collab.total_assistance_requests == 0
	
	def test_page_user_presence(self):
		"""Test user presence management"""
		page_collab = RTCPageCollaboration(
			page_collab_id=uuid7str(),
			tenant_id="tenant123",
			page_url="/admin/users/list",
			page_title="User Management",
			page_type="list_view"
		)
		
		# Add user presence
		page_collab.add_user_presence("user123", {
			"display_name": "John Doe",
			"role": "admin"
		})
		
		assert len(page_collab.current_users) == 1
		assert page_collab.current_users[0] == "user123"
		
		# Remove user presence
		page_collab.remove_user_presence("user123")
		assert len(page_collab.current_users) == 0
	
	def test_form_field_delegation(self):
		"""Test form field delegation"""
		page_collab = RTCPageCollaboration(
			page_collab_id=uuid7str(),
			tenant_id="tenant123",
			page_url="/admin/users/add",
			page_title="Add User",
			page_type="form_view"
		)
		
		# Delegate field
		success = page_collab.delegate_field(
			"email", "user123", "user456", "Please fill in email"
		)
		
		assert success is True
		assert page_collab.total_form_delegations == 1
		assert "email" in page_collab.delegated_fields
		assert page_collab.delegated_fields["email"]["delegatee_id"] == "user456"
	
	def test_assistance_requests(self):
		"""Test assistance request functionality"""
		page_collab = RTCPageCollaboration(
			page_collab_id=uuid7str(),
			tenant_id="tenant123",
			page_url="/admin/users/edit/123",
			page_title="Edit User",
			page_type="form_view"
		)
		
		# Request assistance
		success = page_collab.request_assistance(
			"user123", "password", "How do I reset this field?"
		)
		
		assert success is True
		assert page_collab.total_assistance_requests == 1


class TestRTCThirdPartyIntegration:
	"""Test RTC Third Party Integration model"""
	
	def test_teams_integration(self):
		"""Test Microsoft Teams integration"""
		integration = RTCThirdPartyIntegration(
			integration_id=uuid7str(),
			tenant_id="tenant123",
			platform="teams",
			platform_name="Microsoft Teams",
			integration_type="api",
			teams_tenant_id="teams-tenant-123",
			teams_application_id="app-123"
		)
		
		assert integration.platform == "teams"
		assert integration.teams_tenant_id == "teams-tenant-123"
		assert integration.teams_application_id == "app-123"
		assert integration.status == "active"
	
	def test_zoom_integration(self):
		"""Test Zoom integration"""
		integration = RTCThirdPartyIntegration(
			integration_id=uuid7str(),
			tenant_id="tenant123",
			platform="zoom",
			platform_name="Zoom",
			integration_type="api",
			zoom_account_id="zoom-account-123",
			api_key="zoom-api-key",
			api_secret="zoom-api-secret"
		)
		
		assert integration.platform == "zoom"
		assert integration.zoom_account_id == "zoom-account-123"
		assert integration.api_key == "zoom-api-key"
		assert integration.sync_meetings is True
	
	def test_google_meet_integration(self):
		"""Test Google Meet integration"""
		integration = RTCThirdPartyIntegration(
			integration_id=uuid7str(),
			tenant_id="tenant123",
			platform="google_meet",
			platform_name="Google Meet",
			integration_type="api",
			google_workspace_domain="company.com",
			api_key="google-client-id",
			api_secret="google-client-secret"
		)
		
		assert integration.platform == "google_meet"
		assert integration.google_workspace_domain == "company.com"
		assert integration.api_key == "google-client-id"
		assert integration.auto_create_meetings is False


class TestRTCScreenShare:
	"""Test RTC Screen Share model"""
	
	def test_screen_share_creation(self):
		"""Test creating screen share"""
		screen_share = RTCScreenShare(
			share_id=uuid7str(),
			call_id=uuid7str(),
			presenter_id=uuid7str(),
			tenant_id="tenant123",
			share_type="desktop",
			share_name="Desktop Share"
		)
		
		assert screen_share.share_type == "desktop"
		assert screen_share.share_name == "Desktop Share"
		assert screen_share.status == "active"
		assert screen_share.resolution == "1920x1080"
		assert screen_share.started_at is not None


class TestRTCRecording:
	"""Test RTC Recording model"""
	
	def test_recording_creation(self):
		"""Test creating recording"""
		recording = RTCRecording(
			recording_id=uuid7str(),
			call_id=uuid7str(),
			initiated_by="user123",
			tenant_id="tenant123",
			recording_name="Meeting Recording",
			recording_type="full_meeting"
		)
		
		assert recording.recording_name == "Meeting Recording"
		assert recording.recording_type == "full_meeting"
		assert recording.status == "recording"
		assert recording.auto_transcription_enabled is True
		assert recording.started_at is not None


@pytest.fixture
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


if __name__ == "__main__":
	pytest.main([__file__])
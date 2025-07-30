"""
Unit tests for Real-Time Collaboration service
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid_extensions import uuid7str

from ..service import (
	CollaborationService, CollaborationContext,
	SessionCreateRequest, PageCollaborationRequest,
	FieldDelegationRequest, AssistanceRequest, VideoCallRequest
)
from ..models import RTCSession, RTCVideoCall, RTCPageCollaboration


class TestCollaborationService:
	"""Test Collaboration Service"""
	
	@pytest.fixture
	def mock_db_session(self):
		"""Mock database session"""
		session = AsyncMock()
		session.add = Mock()
		session.commit = AsyncMock()
		session.execute = AsyncMock()
		return session
	
	@pytest.fixture
	def collaboration_context(self):
		"""Sample collaboration context"""
		return CollaborationContext(
			tenant_id="tenant123",
			user_id="user123",
			page_url="/admin/users/list"
		)
	
	@pytest.fixture
	def collaboration_service(self, mock_db_session):
		"""Collaboration service instance"""
		return CollaborationService(mock_db_session)
	
	async def test_create_session(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test creating a collaboration session"""
		session = await collaboration_service.create_session(
			collaboration_context, "Test Session", "page_collaboration"
		)
		
		assert session.session_name == "Test Session"
		assert session.session_type == "page_collaboration"
		assert session.owner_user_id == "user123"
		assert session.tenant_id == "tenant123"
		
		# Verify database operations
		mock_db_session.add.assert_called_once()
		mock_db_session.commit.assert_called_once()
	
	async def test_join_session(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test joining a collaboration session"""
		# Mock session exists
		mock_session = RTCSession(
			session_id=uuid7str(),
			tenant_id="tenant123",
			session_name="Test Session",
			owner_user_id="user123"
		)
		
		mock_result = Mock()
		mock_result.scalar_one_or_none.return_value = mock_session
		mock_db_session.execute.return_value = mock_result
		
		participant = await collaboration_service.join_session(
			collaboration_context, mock_session.session_id, "viewer"
		)
		
		assert participant.user_id == "user123"
		assert participant.role == "viewer"
		assert participant.session_id == mock_session.session_id
		
		# Verify database operations
		mock_db_session.add.assert_called()
		mock_db_session.commit.assert_called()
	
	async def test_join_nonexistent_session(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test joining a nonexistent session raises error"""
		mock_result = Mock()
		mock_result.scalar_one_or_none.return_value = None
		mock_db_session.execute.return_value = mock_result
		
		with pytest.raises(ValueError, match="Session .* not found"):
			await collaboration_service.join_session(
				collaboration_context, "nonexistent-session", "viewer"
			)
	
	async def test_end_session(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test ending a collaboration session"""
		# Mock session exists
		mock_session = RTCSession(
			session_id=uuid7str(),
			tenant_id="tenant123",
			session_name="Test Session",
			owner_user_id="user123",
			actual_start=datetime.utcnow() - timedelta(minutes=30)
		)
		
		mock_result = Mock()
		mock_result.scalar_one_or_none.return_value = mock_session
		mock_db_session.execute.return_value = mock_result
		
		session = await collaboration_service.end_session(
			collaboration_context, mock_session.session_id
		)
		
		assert session.is_active is False
		assert session.actual_end is not None
		assert session.duration_minutes is not None
		
		# Verify database operations
		mock_db_session.commit.assert_called_once()
	
	async def test_enable_page_collaboration(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test enabling page collaboration"""
		# Mock no existing collaboration
		mock_result = Mock()
		mock_result.scalar_one_or_none.return_value = None
		mock_db_session.execute.return_value = mock_result
		
		page_collab = await collaboration_service.enable_page_collaboration(
			collaboration_context, "User Management", "list_view"
		)
		
		assert page_collab.page_url == "/admin/users/list"
		assert page_collab.page_title == "User Management"
		assert page_collab.page_type == "list_view"
		assert page_collab.tenant_id == "tenant123"
		
		# Verify database operations
		mock_db_session.add.assert_called_once()
		mock_db_session.commit.assert_called_once()
	
	@patch('..service.websocket_manager')
	async def test_delegate_form_field(self, mock_websocket_manager, collaboration_service, collaboration_context, mock_db_session):
		"""Test delegating a form field"""
		# Mock page collaboration exists
		mock_page_collab = RTCPageCollaboration(
			page_collab_id=uuid7str(),
			tenant_id="tenant123",
			page_url="/admin/users/list",
			page_title="User Management",
			page_type="list_view"
		)
		
		with patch.object(collaboration_service, '_get_or_create_page_collaboration', return_value=mock_page_collab):
			mock_page_collab.delegate_field = Mock(return_value=True)
			
			success = await collaboration_service.delegate_form_field(
				collaboration_context, "email", "user456", "Please fill this"
			)
			
			assert success is True
			mock_page_collab.delegate_field.assert_called_once_with(
				"email", "user123", "user456", "Please fill this"
			)
			mock_db_session.commit.assert_called_once()
	
	@patch('..service.websocket_manager')
	async def test_request_assistance(self, mock_websocket_manager, collaboration_service, collaboration_context, mock_db_session):
		"""Test requesting assistance"""
		# Mock page collaboration exists
		mock_page_collab = RTCPageCollaboration(
			page_collab_id=uuid7str(),
			tenant_id="tenant123",
			page_url="/admin/users/list",
			page_title="User Management",
			page_type="list_view"
		)
		
		with patch.object(collaboration_service, '_get_or_create_page_collaboration', return_value=mock_page_collab):
			mock_page_collab.request_assistance = Mock(return_value=True)
			
			success = await collaboration_service.request_assistance(
				collaboration_context, "password", "How do I reset this?"
			)
			
			assert success is True
			mock_page_collab.request_assistance.assert_called_once_with(
				"user123", "password", "How do I reset this?"
			)
			mock_db_session.commit.assert_called_once()
	
	@patch('..service.websocket_manager')
	async def test_start_video_call(self, mock_websocket_manager, collaboration_service, collaboration_context, mock_db_session):
		"""Test starting a video call"""
		with patch.object(collaboration_service, '_get_or_create_session') as mock_get_session:
			mock_session = RTCSession(
				session_id=uuid7str(),
				tenant_id="tenant123",
				session_name="Test Session",
				owner_user_id="user123"
			)
			mock_get_session.return_value = mock_session
			
			with patch.object(collaboration_service, '_setup_third_party_integration'):
				video_call = await collaboration_service.start_video_call(
					collaboration_context, "Team Meeting", "video"
				)
				
				assert video_call.call_name == "Team Meeting"
				assert video_call.call_type == "video"
				assert video_call.host_user_id == "user123"
				assert video_call.session_id == mock_session.session_id
				
				# Verify database operations
				mock_db_session.add.assert_called_once()
				mock_db_session.commit.assert_called_once()
	
	async def test_start_screen_share(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test starting screen share"""
		# Mock video call exists
		mock_video_call = RTCVideoCall(
			call_id=uuid7str(),
			session_id=uuid7str(),
			tenant_id="tenant123",
			call_name="Test Call",
			host_user_id="user123"
		)
		
		# Mock video participant exists
		mock_participant = Mock()
		mock_participant.video_participant_id = uuid7str()
		
		mock_result1 = Mock()
		mock_result1.scalar_one_or_none.return_value = mock_video_call
		
		mock_result2 = Mock()
		mock_result2.scalar_one_or_none.return_value = mock_participant
		
		mock_db_session.execute.side_effect = [mock_result1, mock_result2]
		
		screen_share = await collaboration_service.start_screen_share(
			collaboration_context, mock_video_call.call_id, "desktop", "My Desktop"
		)
		
		assert screen_share.call_id == mock_video_call.call_id
		assert screen_share.share_type == "desktop"
		assert screen_share.share_name == "My Desktop"
		
		# Verify database operations
		mock_db_session.add.assert_called_once()
		mock_db_session.commit.assert_called_once()
	
	async def test_start_recording(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test starting a recording"""
		recording = await collaboration_service.start_recording(
			collaboration_context, "call123", "Meeting Recording", "full_meeting"
		)
		
		assert recording.call_id == "call123"
		assert recording.recording_name == "Meeting Recording"
		assert recording.recording_type == "full_meeting"
		assert recording.initiated_by == "user123"
		
		# Verify database operations
		mock_db_session.add.assert_called_once()
		mock_db_session.commit.assert_called_once()
	
	async def test_setup_teams_integration(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test setting up Teams integration"""
		integration = await collaboration_service.setup_teams_integration(
			collaboration_context, "teams-tenant-123", "app-123"
		)
		
		assert integration.platform == "teams"
		assert integration.teams_tenant_id == "teams-tenant-123"
		assert integration.teams_application_id == "app-123"
		assert integration.tenant_id == "tenant123"
		
		# Verify database operations
		mock_db_session.add.assert_called_once()
		mock_db_session.commit.assert_called_once()
	
	async def test_setup_zoom_integration(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test setting up Zoom integration"""
		integration = await collaboration_service.setup_zoom_integration(
			collaboration_context, "zoom-account-123", "api-key", "api-secret"
		)
		
		assert integration.platform == "zoom"
		assert integration.zoom_account_id == "zoom-account-123"
		assert integration.api_key == "api-key"
		assert integration.api_secret == "api-secret"
		
		# Verify database operations
		mock_db_session.add.assert_called_once()
		mock_db_session.commit.assert_called_once()
	
	async def test_setup_google_meet_integration(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test setting up Google Meet integration"""
		integration = await collaboration_service.setup_google_meet_integration(
			collaboration_context, "company.com", "client-id", "client-secret"
		)
		
		assert integration.platform == "google_meet"
		assert integration.google_workspace_domain == "company.com"
		assert integration.api_key == "client-id"
		assert integration.api_secret == "client-secret"
		
		# Verify database operations
		mock_db_session.add.assert_called_once()
		mock_db_session.commit.assert_called_once()
	
	async def test_get_collaboration_analytics(self, collaboration_service, collaboration_context, mock_db_session):
		"""Test getting collaboration analytics"""
		# Mock database results
		mock_result1 = Mock()
		mock_result1.scalars.return_value.all.return_value = []
		
		mock_result2 = Mock()
		mock_result2.scalars.return_value.all.return_value = []
		
		mock_db_session.execute.side_effect = [mock_result1, mock_result2]
		
		with patch('..service.websocket_manager') as mock_websocket_manager:
			mock_websocket_manager.get_connection_stats.return_value = {}
			
			analytics = await collaboration_service.get_collaboration_analytics(collaboration_context)
			
			assert 'date_range' in analytics
			assert 'page_collaboration' in analytics
			assert 'sessions' in analytics
			assert 'websocket_stats' in analytics
	
	def test_extract_blueprint_name(self, collaboration_service):
		"""Test extracting blueprint name from URL"""
		assert collaboration_service._extract_blueprint_name("/admin/users/list") == "admin"
		assert collaboration_service._extract_blueprint_name("/crm/accounts/") == "crm"
		assert collaboration_service._extract_blueprint_name("/") == "unknown"
	
	def test_extract_view_name(self, collaboration_service):
		"""Test extracting view name from URL"""
		assert collaboration_service._extract_view_name("/admin/users/list") == "users"
		assert collaboration_service._extract_view_name("/crm/accounts/add") == "accounts"
		assert collaboration_service._extract_view_name("/single/") == "unknown"
	
	def test_generate_meeting_id(self, collaboration_service):
		"""Test meeting ID generation"""
		meeting_id = collaboration_service._generate_meeting_id()
		assert len(meeting_id) == 10
		assert meeting_id.isdigit()


class TestPydanticModels:
	"""Test Pydantic request models"""
	
	def test_session_create_request(self):
		"""Test SessionCreateRequest validation"""
		request = SessionCreateRequest(
			session_name="Test Session",
			session_type="page_collaboration",
			page_url="/admin/users/list"
		)
		
		assert request.session_name == "Test Session"
		assert request.session_type == "page_collaboration"
		assert request.page_url == "/admin/users/list"
	
	def test_session_create_request_validation(self):
		"""Test SessionCreateRequest validation errors"""
		with pytest.raises(ValueError):
			SessionCreateRequest(
				session_name="",  # Too short
				page_url="/admin/users/list"
			)
		
		with pytest.raises(ValueError):
			SessionCreateRequest(
				session_name="Test Session",
				page_url=""  # Too short
			)
	
	def test_page_collaboration_request(self):
		"""Test PageCollaborationRequest validation"""
		request = PageCollaborationRequest(
			page_url="/admin/users/list",
			page_title="User Management",
			page_type="list_view"
		)
		
		assert request.page_url == "/admin/users/list"
		assert request.page_title == "User Management"
		assert request.page_type == "list_view"
	
	def test_field_delegation_request(self):
		"""Test FieldDelegationRequest validation"""
		request = FieldDelegationRequest(
			field_name="email",
			delegatee_id="user456",
			instructions="Please fill in the email address"
		)
		
		assert request.field_name == "email"
		assert request.delegatee_id == "user456"
		assert request.instructions == "Please fill in the email address"
	
	def test_assistance_request(self):
		"""Test AssistanceRequest validation"""
		request = AssistanceRequest(
			field_name="password",
			description="How do I reset this field?"
		)
		
		assert request.field_name == "password"
		assert request.description == "How do I reset this field?"
	
	def test_video_call_request(self):
		"""Test VideoCallRequest validation"""
		request = VideoCallRequest(
			call_name="Team Meeting",
			call_type="video",
			enable_recording=True
		)
		
		assert request.call_name == "Team Meeting"
		assert request.call_type == "video"
		assert request.enable_recording is True


@pytest.fixture
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


if __name__ == "__main__":
	pytest.main([__file__])
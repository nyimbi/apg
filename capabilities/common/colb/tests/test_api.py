"""
Integration tests for Real-Time Collaboration API endpoints
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid_extensions import uuid7str

from fastapi.testclient import TestClient
from fastapi import FastAPI

from ..api import router
from ..models import RTCSession, RTCVideoCall, RTCPageCollaboration


# Create test app
app = FastAPI()
app.include_router(router)


class TestSessionAPI:
	"""Test session management API endpoints"""
	
	@pytest.fixture
	def client(self):
		"""Create test client"""
		return TestClient(app)
	
	@pytest.fixture
	def mock_user(self):
		"""Mock current user"""
		return {
			"user_id": "user123",
			"tenant_id": "tenant123",
			"username": "testuser",
			"permissions": ["rtc:*"]
		}
	
	def test_create_session(self, client):
		"""Test creating a new session"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123",
				"username": "testuser"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_session = RTCSession(
				session_id=uuid7str(),
				tenant_id="tenant123",
				session_name="Test Session",
				session_type="page_collaboration",
				owner_user_id="user123",
				actual_start=datetime.utcnow(),
				current_participant_count=0
			)
			
			mock_service_instance = AsyncMock()
			mock_service_instance.create_session.return_value = mock_session
			mock_service.return_value = mock_service_instance
			
			# Make request
			response = client.post("/api/v1/rtc/sessions", json={
				"session_name": "Test Session",
				"session_type": "page_collaboration",
				"page_url": "/admin/users/list"
			})
			
			assert response.status_code == 200
			data = response.json()
			assert data["session_name"] == "Test Session"
			assert data["session_type"] == "page_collaboration"
			assert data["owner_user_id"] == "user123"
	
	def test_join_session(self, client):
		"""Test joining a session"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_participant = Mock()
			mock_participant.participant_id = uuid7str()
			mock_participant.role = "viewer"
			mock_participant.joined_at = datetime.utcnow()
			mock_participant.can_edit = False
			mock_participant.can_annotate = True
			mock_participant.can_chat = True
			mock_participant.can_share_screen = False
			
			mock_service_instance = AsyncMock()
			mock_service_instance.join_session.return_value = mock_participant
			mock_service.return_value = mock_service_instance
			
			# Make request
			session_id = uuid7str()
			response = client.post(f"/api/v1/rtc/sessions/{session_id}/join?role=viewer")
			
			assert response.status_code == 200
			data = response.json()
			assert data["session_id"] == session_id
			assert data["role"] == "viewer"
			assert "permissions" in data
	
	def test_get_session(self, client):
		"""Test getting session details"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db:
			
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			session_id = uuid7str()
			response = client.get(f"/api/v1/rtc/sessions/{session_id}")
			
			assert response.status_code == 200
			data = response.json()
			assert data["session_id"] == session_id


class TestPageCollaborationAPI:
	"""Test page collaboration API endpoints"""
	
	@pytest.fixture
	def client(self):
		"""Create test client"""
		return TestClient(app)
	
	def test_enable_page_collaboration(self, client):
		"""Test enabling page collaboration"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_page_collab = RTCPageCollaboration(
				page_collab_id=uuid7str(),
				tenant_id="tenant123",
				page_url="/admin/users/list",
				page_title="User Management",
				page_type="list_view",
				current_users=["user123"],
				is_active=True,
				total_form_delegations=0,
				total_assistance_requests=0
			)
			
			mock_service_instance = AsyncMock()
			mock_service_instance.enable_page_collaboration.return_value = mock_page_collab
			mock_service.return_value = mock_service_instance
			
			# Make request
			response = client.post("/api/v1/rtc/page-collaboration", json={
				"page_url": "/admin/users/list",
				"page_title": "User Management",
				"page_type": "list_view"
			})
			
			assert response.status_code == 200
			data = response.json()
			assert data["page_url"] == "/admin/users/list"
			assert data["page_title"] == "User Management"
			assert data["is_active"] is True
	
	def test_delegate_form_field(self, client):
		"""Test delegating a form field"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_service_instance = AsyncMock()
			mock_service_instance.delegate_form_field.return_value = True
			mock_service.return_value = mock_service_instance
			
			# Make request
			response = client.post(
				"/api/v1/rtc/page-collaboration/delegate-field?page_url=/admin/users/add",
				json={
					"field_name": "email",
					"delegatee_id": "user456",
					"instructions": "Please fill this field"
				}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["field_name"] == "email"
			assert "delegated successfully" in data["message"]
	
	def test_request_assistance(self, client):
		"""Test requesting assistance"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_service_instance = AsyncMock()
			mock_service_instance.request_assistance.return_value = True
			mock_service.return_value = mock_service_instance
			
			# Make request
			response = client.post(
				"/api/v1/rtc/page-collaboration/request-assistance?page_url=/admin/users/edit/123",
				json={
					"field_name": "password",
					"description": "How do I reset this field?"
				}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert "requested successfully" in data["message"]
	
	def test_get_page_presence(self, client):
		"""Test getting page presence"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.websocket_manager') as mock_websocket_manager:
			
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123",
				"username": "testuser"
			}
			
			mock_websocket_manager.get_connection_stats.return_value = {}
			
			# Make request
			response = client.get("/api/v1/rtc/page-collaboration/presence?page_url=/admin/users/list")
			
			assert response.status_code == 200
			data = response.json()
			assert isinstance(data, list)
			assert len(data) == 1  # Current user
			assert data[0]["user_id"] == "user123"


class TestVideoCallAPI:
	"""Test video call API endpoints"""
	
	@pytest.fixture
	def client(self):
		"""Create test client"""
		return TestClient(app)
	
	def test_start_video_call(self, client):
		"""Test starting a video call"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_video_call = RTCVideoCall(
				call_id=uuid7str(),
				session_id=uuid7str(),
				tenant_id="tenant123",
				call_name="Team Meeting",
				call_type="video",
				host_user_id="user123",
				meeting_id="123456789",
				teams_meeting_url=None,
				zoom_meeting_id=None,
				meet_url=None,
				current_participants=1,
				max_participants=100,
				enable_recording=False
			)
			
			mock_service_instance = AsyncMock()
			mock_service_instance.start_video_call.return_value = mock_video_call
			mock_service.return_value = mock_service_instance
			
			# Make request
			response = client.post(
				"/api/v1/rtc/video-calls?page_url=/admin/dashboard",
				json={
					"call_name": "Team Meeting",
					"call_type": "video",
					"enable_recording": False
				}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["call_name"] == "Team Meeting"
			assert data["call_type"] == "video"
			assert data["host_user_id"] == "user123"
	
	def test_start_screen_share(self, client):
		"""Test starting screen sharing"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_screen_share = Mock()
			mock_screen_share.share_id = uuid7str()
			mock_screen_share.share_type = "desktop"
			mock_screen_share.status = "active"
			mock_screen_share.started_at = datetime.utcnow()
			
			mock_service_instance = AsyncMock()
			mock_service_instance.start_screen_share.return_value = mock_screen_share
			mock_service.return_value = mock_service_instance
			
			# Make request
			call_id = uuid7str()
			response = client.post(f"/api/v1/rtc/video-calls/{call_id}/screen-share?share_type=desktop")
			
			assert response.status_code == 200
			data = response.json()
			assert data["call_id"] == call_id
			assert data["share_type"] == "desktop"
			assert data["presenter_id"] == "user123"
	
	def test_start_recording(self, client):
		"""Test starting recording"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_recording = Mock()
			mock_recording.recording_id = uuid7str()
			mock_recording.recording_name = "Meeting Recording"
			mock_recording.recording_type = "full_meeting"
			mock_recording.status = "recording"
			mock_recording.started_at = datetime.utcnow()
			mock_recording.auto_transcription_enabled = True
			
			mock_service_instance = AsyncMock()
			mock_service_instance.start_recording.return_value = mock_recording
			mock_service.return_value = mock_service_instance
			
			# Make request
			call_id = uuid7str()
			response = client.post(
				f"/api/v1/rtc/video-calls/{call_id}/recording?recording_name=Meeting Recording&recording_type=full_meeting"
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["call_id"] == call_id
			assert data["recording_name"] == "Meeting Recording"
			assert data["auto_transcription"] is True
	
	def test_join_video_call(self, client):
		"""Test joining a video call"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db:
			
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			call_id = uuid7str()
			response = client.post(f"/api/v1/rtc/video-calls/{call_id}/participants")
			
			assert response.status_code == 200
			data = response.json()
			assert data["call_id"] == call_id
			assert data["user_id"] == "user123"
			assert data["role"] == "attendee"
			assert "permissions" in data


class TestThirdPartyIntegrationAPI:
	"""Test third-party integration API endpoints"""
	
	@pytest.fixture
	def client(self):
		"""Create test client"""
		return TestClient(app)
	
	def test_setup_teams_integration(self, client):
		"""Test setting up Teams integration"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_integration = Mock()
			mock_integration.integration_id = uuid7str()
			mock_integration.platform = "teams"
			mock_integration.status = "active"
			mock_integration.teams_tenant_id = "teams-tenant-123"
			mock_integration.created_at = datetime.utcnow()
			
			mock_service_instance = AsyncMock()
			mock_service_instance.setup_teams_integration.return_value = mock_integration
			mock_service.return_value = mock_service_instance
			
			# Make request
			response = client.post(
				"/api/v1/rtc/integrations/teams?teams_tenant_id=teams-tenant-123&application_id=app-123"
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["platform"] == "teams"
			assert data["teams_tenant_id"] == "teams-tenant-123"
			assert data["status"] == "active"
	
	def test_setup_zoom_integration(self, client):
		"""Test setting up Zoom integration"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_integration = Mock()
			mock_integration.integration_id = uuid7str()
			mock_integration.platform = "zoom"
			mock_integration.status = "active"
			mock_integration.zoom_account_id = "zoom-account-123"
			mock_integration.created_at = datetime.utcnow()
			
			mock_service_instance = AsyncMock()
			mock_service_instance.setup_zoom_integration.return_value = mock_integration
			mock_service.return_value = mock_service_instance
			
			# Make request
			response = client.post(
				"/api/v1/rtc/integrations/zoom?zoom_account_id=zoom-account-123&api_key=key&api_secret=secret"
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["platform"] == "zoom"
			assert data["zoom_account_id"] == "zoom-account-123"


class TestChatAPI:
	"""Test chat and messaging API endpoints"""
	
	@pytest.fixture
	def client(self):
		"""Create test client"""
		return TestClient(app)
	
	def test_send_chat_message(self, client):
		"""Test sending a chat message"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.websocket_manager') as mock_websocket_manager:
			
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123",
				"username": "testuser"
			}
			
			mock_websocket_manager._broadcast_to_page = AsyncMock()
			
			# Make request
			response = client.post(
				"/api/v1/rtc/chat/messages?message=Hello everyone&page_url=/admin/users/list&message_type=text"
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["message"] == "Hello everyone"
			assert "message_id" in data
			assert "sent_at" in data
	
	def test_get_chat_messages(self, client):
		"""Test getting chat messages"""
		with patch('..api.get_current_user') as mock_get_user:
			
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			
			# Make request
			response = client.get("/api/v1/rtc/chat/messages?page_url=/admin/users/list&limit=50")
			
			assert response.status_code == 200
			data = response.json()
			assert "messages" in data
			assert data["page_url"] == "/admin/users/list"
			assert data["total_count"] == 0


class TestAnalyticsAPI:
	"""Test analytics API endpoints"""
	
	@pytest.fixture
	def client(self):
		"""Create test client"""
		return TestClient(app)
	
	def test_get_collaboration_analytics(self, client):
		"""Test getting collaboration analytics"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.get_async_session') as mock_get_db, \
			 patch('..api.CollaborationService') as mock_service:
			
			# Setup mocks
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			mock_get_db.return_value = AsyncMock()
			
			mock_analytics = {
				"date_range": {
					"start": "2024-01-01T00:00:00",
					"end": "2024-01-31T23:59:59"
				},
				"page_collaboration": {
					"total_pages": 15,
					"total_delegations": 25,
					"total_assistance_requests": 8
				},
				"sessions": {
					"total_sessions": 42,
					"active_sessions": 3,
					"average_duration": 35.5
				}
			}
			
			mock_service_instance = AsyncMock()
			mock_service_instance.get_collaboration_analytics.return_value = mock_analytics
			mock_service.return_value = mock_service_instance
			
			# Make request
			response = client.get("/api/v1/rtc/analytics")
			
			assert response.status_code == 200
			data = response.json()
			assert "date_range" in data
			assert "page_collaboration" in data
			assert "sessions" in data
	
	def test_get_presence_analytics(self, client):
		"""Test getting presence analytics"""
		with patch('..api.get_current_user') as mock_get_user, \
			 patch('..api.websocket_manager') as mock_websocket_manager:
			
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			
			mock_websocket_manager.get_connection_stats.return_value = {
				"total_connections": 15,
				"unique_users": 12,
				"unique_pages": 8
			}
			
			# Make request
			response = client.get("/api/v1/rtc/analytics/presence")
			
			assert response.status_code == 200
			data = response.json()
			assert "realtime_stats" in data
			assert "timestamp" in data
			assert data["tenant_id"] == "tenant123"


class TestHealthAPI:
	"""Test health and status API endpoints"""
	
	@pytest.fixture
	def client(self):
		"""Create test client"""
		return TestClient(app)
	
	def test_health_check(self, client):
		"""Test health check endpoint"""
		with patch('..api.websocket_manager') as mock_websocket_manager:
			
			mock_websocket_manager.get_connection_stats.return_value = {
				"total_connections": 5,
				"unique_users": 3
			}
			
			# Make request
			response = client.get("/api/v1/rtc/health")
			
			assert response.status_code == 200
			data = response.json()
			assert data["status"] == "healthy"
			assert "timestamp" in data
			assert "websocket_stats" in data
			assert data["version"] == "1.0.0"
	
	def test_get_status(self, client):
		"""Test getting service status"""
		with patch('..api.get_current_user') as mock_get_user:
			
			mock_get_user.return_value = {
				"user_id": "user123",
				"tenant_id": "tenant123"
			}
			
			# Make request
			response = client.get("/api/v1/rtc/status")
			
			assert response.status_code == 200
			data = response.json()
			assert data["service"] == "real-time-collaboration"
			assert data["version"] == "1.0.0"
			assert "features" in data
			assert "limits" in data
			assert data["features"]["page_collaboration"] is True
			assert data["features"]["teams_integration"] is True


@pytest.fixture
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


if __name__ == "__main__":
	pytest.main([__file__])
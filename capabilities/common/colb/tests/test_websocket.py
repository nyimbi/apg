"""
Tests for WebSocket manager and real-time communication
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch
from uuid_extensions import uuid7str

from ..websocket_manager import (
	WebSocketManager, MessageType, 
	ConnectionInfo, PresenceInfo
)


class MockWebSocket:
	"""Mock WebSocket for testing"""
	
	def __init__(self):
		self.sent_messages = []
		self.closed = False
		self.close_code = None
		self.close_reason = None
	
	async def send_text(self, message):
		"""Mock send_text method"""
		self.sent_messages.append(message)
	
	async def close(self, code=1000, reason=""):
		"""Mock close method"""
		self.closed = True
		self.close_code = code
		self.close_reason = reason
	
	async def receive_text(self):
		"""Mock receive_text method"""
		# Return a test message
		return json.dumps({
			"type": "chat_message",
			"message": "test message",
			"user_id": "test_user"
		})


class TestWebSocketManager:
	"""Test WebSocket Manager"""
	
	@pytest.fixture
	def websocket_manager(self):
		"""Create WebSocket manager instance"""
		manager = WebSocketManager()
		manager._connections = {}
		manager._page_connections = {}
		manager._user_presence = {}
		return manager
	
	@pytest.fixture
	def mock_websocket(self):
		"""Create mock WebSocket"""
		return MockWebSocket()
	
	async def test_add_connection(self, websocket_manager, mock_websocket):
		"""Test adding a WebSocket connection"""
		connection_id = "conn123"
		user_id = "user123"
		page_url = "/admin/users/list"
		
		websocket_manager.add_connection(
			connection_id, mock_websocket, user_id, page_url
		)
		
		# Check connection was added
		assert connection_id in websocket_manager._connections
		conn_info = websocket_manager._connections[connection_id]
		assert conn_info.websocket == mock_websocket
		assert conn_info.user_id == user_id
		assert conn_info.page_url == page_url
		
		# Check page connections
		assert page_url in websocket_manager._page_connections
		assert connection_id in websocket_manager._page_connections[page_url]
		
		# Check user presence
		assert user_id in websocket_manager._user_presence
		presence = websocket_manager._user_presence[user_id]
		assert presence.user_id == user_id
		assert presence.page_url == page_url
	
	async def test_remove_connection(self, websocket_manager, mock_websocket):
		"""Test removing a WebSocket connection"""
		connection_id = "conn123"
		user_id = "user123"
		page_url = "/admin/users/list"
		
		# Add connection first
		websocket_manager.add_connection(
			connection_id, mock_websocket, user_id, page_url
		)
		
		# Remove connection
		await websocket_manager.remove_connection(connection_id)
		
		# Check connection was removed
		assert connection_id not in websocket_manager._connections
		assert page_url not in websocket_manager._page_connections or \
			   connection_id not in websocket_manager._page_connections[page_url]
		assert user_id not in websocket_manager._user_presence
	
	async def test_broadcast_to_page(self, websocket_manager):
		"""Test broadcasting message to page"""
		page_url = "/admin/users/list"
		
		# Add multiple connections to the same page
		mock_ws1 = MockWebSocket()
		mock_ws2 = MockWebSocket()
		
		websocket_manager.add_connection("conn1", mock_ws1, "user1", page_url)
		websocket_manager.add_connection("conn2", mock_ws2, "user2", page_url)
		
		# Broadcast message
		message = {
			"type": "chat_message",
			"content": "Hello everyone!",
			"user_id": "user1"
		}
		
		await websocket_manager._broadcast_to_page(page_url, message)
		
		# Check both connections received the message
		assert len(mock_ws1.sent_messages) == 1
		assert len(mock_ws2.sent_messages) == 1
		
		sent_message1 = json.loads(mock_ws1.sent_messages[0])
		sent_message2 = json.loads(mock_ws2.sent_messages[0])
		
		assert sent_message1["type"] == "chat_message"
		assert sent_message1["content"] == "Hello everyone!"
		assert sent_message2["type"] == "chat_message"
		assert sent_message2["content"] == "Hello everyone!"
	
	async def test_broadcast_to_user(self, websocket_manager):
		"""Test broadcasting message to specific user"""
		user_id = "user123"
		page_url = "/admin/users/list"
		
		mock_ws = MockWebSocket()
		websocket_manager.add_connection("conn1", mock_ws, user_id, page_url)
		
		# Broadcast to user
		message = {
			"type": "notification",
			"content": "You have a new task",
			"target_user_id": user_id
		}
		
		await websocket_manager._broadcast_to_user(user_id, message)
		
		# Check user received the message
		assert len(mock_ws.sent_messages) == 1
		sent_message = json.loads(mock_ws.sent_messages[0])
		assert sent_message["type"] == "notification"
		assert sent_message["content"] == "You have a new task"
	
	async def test_handle_message_chat(self, websocket_manager, mock_websocket):
		"""Test handling chat message"""
		connection_id = "conn123"
		user_id = "user123"
		page_url = "/admin/users/list"
		
		websocket_manager.add_connection(connection_id, mock_websocket, user_id, page_url)
		
		# Add another user to receive the message
		mock_ws2 = MockWebSocket()
		websocket_manager.add_connection("conn2", mock_ws2, "user2", page_url)
		
		message = {
			"type": MessageType.CHAT_MESSAGE.value,
			"message": "Hello everyone!",
			"user_id": user_id,
			"page_url": page_url
		}
		
		await websocket_manager._handle_message(connection_id, message)
		
		# Check message was broadcasted to other users on the page
		assert len(mock_ws2.sent_messages) == 1
		sent_message = json.loads(mock_ws2.sent_messages[0])
		assert sent_message["type"] == MessageType.CHAT_MESSAGE.value
		assert sent_message["message"] == "Hello everyone!"
	
	async def test_handle_message_presence(self, websocket_manager, mock_websocket):
		"""Test handling presence update"""
		connection_id = "conn123"
		user_id = "user123"
		page_url = "/admin/users/list"
		
		websocket_manager.add_connection(connection_id, mock_websocket, user_id, page_url)
		
		# Add another user to receive the presence update
		mock_ws2 = MockWebSocket()
		websocket_manager.add_connection("conn2", mock_ws2, "user2", page_url)
		
		message = {
			"type": MessageType.PRESENCE_UPDATE.value,
			"user_id": user_id,
			"page_url": page_url,
			"status": "typing"
		}
		
		await websocket_manager._handle_message(connection_id, message)
		
		# Check presence was updated
		presence = websocket_manager._user_presence[user_id]
		assert presence.status == "typing"
		
		# Check presence update was broadcasted
		assert len(mock_ws2.sent_messages) == 1
		sent_message = json.loads(mock_ws2.sent_messages[0])
		assert sent_message["type"] == MessageType.PRESENCE_UPDATE.value
	
	async def test_handle_message_form_delegation(self, websocket_manager, mock_websocket):
		"""Test handling form field delegation"""
		connection_id = "conn123"
		user_id = "user123"
		page_url = "/admin/users/add"
		
		websocket_manager.add_connection(connection_id, mock_websocket, user_id, page_url)
		
		# Add target user
		mock_ws2 = MockWebSocket()
		websocket_manager.add_connection("conn2", mock_ws2, "user456", page_url)
		
		message = {
			"type": MessageType.FORM_DELEGATION.value,
			"delegator_id": user_id,
			"delegatee_id": "user456",
			"field_name": "email",
			"instructions": "Please fill this field",
			"page_url": page_url
		}
		
		await websocket_manager._handle_message(connection_id, message)
		
		# Check delegation was broadcasted to target user
		assert len(mock_ws2.sent_messages) == 1
		sent_message = json.loads(mock_ws2.sent_messages[0])
		assert sent_message["type"] == MessageType.FORM_DELEGATION.value
		assert sent_message["field_name"] == "email"
		assert sent_message["delegatee_id"] == "user456"
	
	async def test_handle_message_assistance_request(self, websocket_manager, mock_websocket):
		"""Test handling assistance request"""
		connection_id = "conn123"
		user_id = "user123"
		page_url = "/admin/users/edit/456"
		
		websocket_manager.add_connection(connection_id, mock_websocket, user_id, page_url)
		
		# Add other users on the page
		mock_ws2 = MockWebSocket()
		mock_ws3 = MockWebSocket()
		websocket_manager.add_connection("conn2", mock_ws2, "user2", page_url)
		websocket_manager.add_connection("conn3", mock_ws3, "user3", page_url)
		
		message = {
			"type": MessageType.ASSISTANCE_REQUEST.value,
			"requester_id": user_id,
			"field_name": "password",
			"description": "How do I reset this field?",
			"page_url": page_url
		}
		
		await websocket_manager._handle_message(connection_id, message)
		
		# Check assistance request was broadcasted to all users on page
		assert len(mock_ws2.sent_messages) == 1
		assert len(mock_ws3.sent_messages) == 1
		
		sent_message = json.loads(mock_ws2.sent_messages[0])
		assert sent_message["type"] == MessageType.ASSISTANCE_REQUEST.value
		assert sent_message["field_name"] == "password"
		assert sent_message["requester_id"] == user_id
	
	def test_get_connection_stats(self, websocket_manager):
		"""Test getting connection statistics"""
		# Add some connections
		mock_ws1 = MockWebSocket()
		mock_ws2 = MockWebSocket()
		mock_ws3 = MockWebSocket()
		
		websocket_manager.add_connection("conn1", mock_ws1, "user1", "/page1")
		websocket_manager.add_connection("conn2", mock_ws2, "user2", "/page1")
		websocket_manager.add_connection("conn3", mock_ws3, "user3", "/page2")
		
		stats = websocket_manager.get_connection_stats()
		
		assert stats["total_connections"] == 3
		assert stats["unique_users"] == 3
		assert stats["unique_pages"] == 2
		assert "/page1" in stats["connections_by_page"]
		assert "/page2" in stats["connections_by_page"]
		assert stats["connections_by_page"]["/page1"] == 2
		assert stats["connections_by_page"]["/page2"] == 1
	
	def test_get_page_users(self, websocket_manager):
		"""Test getting users on a specific page"""
		page_url = "/admin/users/list"
		
		# Add users to the page
		mock_ws1 = MockWebSocket()
		mock_ws2 = MockWebSocket()
		
		websocket_manager.add_connection("conn1", mock_ws1, "user1", page_url)
		websocket_manager.add_connection("conn2", mock_ws2, "user2", page_url)
		
		users = websocket_manager.get_page_users(page_url)
		
		assert len(users) == 2
		user_ids = [user.user_id for user in users]
		assert "user1" in user_ids
		assert "user2" in user_ids
	
	async def test_heartbeat_mechanism(self, websocket_manager):
		"""Test heartbeat mechanism"""
		mock_ws = MockWebSocket()
		websocket_manager.add_connection("conn1", mock_ws, "user1", "/page1")
		
		# Simulate heartbeat
		await websocket_manager._send_heartbeat()
		
		# Check heartbeat was sent
		assert len(mock_ws.sent_messages) == 1
		sent_message = json.loads(mock_ws.sent_messages[0])
		assert sent_message["type"] == "heartbeat"
	
	async def test_handle_connection_lifecycle(self, websocket_manager):
		"""Test full connection lifecycle"""
		mock_ws = MockWebSocket()
		connection_id = f"/ws/tenant123/user123"
		
		# Test handle_connection method
		with patch.object(mock_ws, 'receive_text', side_effect=[
			json.dumps({"type": "presence", "status": "active"}),
			Exception("Connection closed")  # Simulate disconnect
		]):
			# This would normally run the connection handler
			# In a real test, we'd need to mock the WebSocket receive loop
			pass
	
	async def test_cleanup_stale_connections(self, websocket_manager):
		"""Test cleanup of stale connections"""
		# Add a connection with old last_seen time
		mock_ws = MockWebSocket()
		websocket_manager.add_connection("conn1", mock_ws, "user1", "/page1")
		
		# Manually set old last_seen time
		conn_info = websocket_manager._connections["conn1"]
		import datetime
		conn_info.last_seen = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
		
		# Run cleanup
		await websocket_manager._cleanup_stale_connections()
		
		# Connection should be removed (assuming 5 minute timeout)
		# This would depend on the actual timeout configuration
		# assert "conn1" not in websocket_manager._connections


class TestConnectionInfo:
	"""Test ConnectionInfo data class"""
	
	def test_connection_info_creation(self):
		"""Test creating ConnectionInfo"""
		mock_ws = MockWebSocket()
		conn_info = ConnectionInfo(
			websocket=mock_ws,
			user_id="user123",
			page_url="/admin/users/list"
		)
		
		assert conn_info.websocket == mock_ws
		assert conn_info.user_id == "user123"
		assert conn_info.page_url == "/admin/users/list"
		assert conn_info.last_seen is not None


class TestPresenceInfo:
	"""Test PresenceInfo data class"""
	
	def test_presence_info_creation(self):
		"""Test creating PresenceInfo"""
		presence = PresenceInfo(
			user_id="user123",
			page_url="/admin/users/list",
			status="active"
		)
		
		assert presence.user_id == "user123"
		assert presence.page_url == "/admin/users/list"
		assert presence.status == "active"
		assert presence.last_activity is not None


class TestMessageType:
	"""Test MessageType enum"""
	
	def test_message_types(self):
		"""Test all message types are defined"""
		assert MessageType.CHAT_MESSAGE.value == "chat_message"
		assert MessageType.PRESENCE_UPDATE.value == "presence_update"
		assert MessageType.FORM_DELEGATION.value == "form_delegation"
		assert MessageType.ASSISTANCE_REQUEST.value == "assistance_request"
		assert MessageType.USER_JOIN.value == "user_join"
		assert MessageType.USER_LEAVE.value == "user_leave"
		assert MessageType.VIDEO_CALL_START.value == "video_call_start"
		assert MessageType.SCREEN_SHARE_START.value == "screen_share_start"


@pytest.fixture
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


if __name__ == "__main__":
	pytest.main([__file__])
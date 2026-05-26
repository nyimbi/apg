"""Focused tests for executable CONN notification authentication."""

import json

import pytest

from capabilities.common.conn.notifications import (
	NotificationType,
	WebSocketClient,
	WebSocketNotificationServer,
	validate_notification_authentication
)
from capabilities.common.conn.security import AuthenticationError, AuthenticationManager


class FakeWebSocket:
	def __init__(self):
		self.sent_messages = []

	async def send(self, message):
		self.sent_messages.append(json.loads(message))


def _auth_manager_with_user():
	manager = AuthenticationManager()
	user = manager.create_user(
		username="notification_user",
		email="notification@example.com",
		password="Str0ng!Signal2026",
		tenant_id="tenant-a",
		roles=["viewer"]
	)
	return manager, user


def test_notification_authentication_accepts_jwt_claims():
	manager, user = _auth_manager_with_user()
	token = manager.generate_jwt_token(user)

	result = validate_notification_authentication(
		token=token,
		user_id=user.user_id,
		tenant_id=user.tenant_id,
		authentication_manager=manager
	)

	assert result.user_id == user.user_id
	assert result.tenant_id == "tenant-a"
	assert result.source == "jwt"


def test_notification_authentication_rejects_claim_mismatch():
	manager, user = _auth_manager_with_user()
	token = manager.generate_jwt_token(user)

	with pytest.raises(AuthenticationError, match="Notification tenant does not match JWT principal"):
		validate_notification_authentication(
			token=token,
			user_id=user.user_id,
			tenant_id="tenant-b",
			authentication_manager=manager
		)


@pytest.mark.asyncio
async def test_websocket_authentication_sets_identity_from_valid_token():
	manager, user = _auth_manager_with_user()
	token = manager.generate_jwt_token(user)
	server = WebSocketNotificationServer(authentication_manager=manager)
	websocket = FakeWebSocket()
	connection_id = "client-1"
	server.clients[connection_id] = WebSocketClient(connection_id, websocket)

	await server._authenticate_client(connection_id, {
		"token": token,
		"user_id": user.user_id,
		"tenant_id": user.tenant_id
	})

	client = server.clients[connection_id]
	assert client.user_id == user.user_id
	assert client.tenant_id == "tenant-a"
	assert client.metadata["auth_source"] == "jwt"
	assert websocket.sent_messages[-1]["notification"]["title"] == "Authenticated"


@pytest.mark.asyncio
async def test_websocket_authentication_rejects_invalid_token():
	manager, user = _auth_manager_with_user()
	server = WebSocketNotificationServer(authentication_manager=manager)
	websocket = FakeWebSocket()
	connection_id = "client-1"
	server.clients[connection_id] = WebSocketClient(connection_id, websocket)

	await server._authenticate_client(connection_id, {
		"token": "not-a-valid-token",
		"user_id": user.user_id,
		"tenant_id": user.tenant_id
	})

	client = server.clients[connection_id]
	assert client.user_id is None
	assert client.tenant_id is None
	assert websocket.sent_messages[-1]["notification"]["type"] == NotificationType.SECURITY_EVENT.value
	assert websocket.sent_messages[-1]["notification"]["title"] == "Authentication Failed"

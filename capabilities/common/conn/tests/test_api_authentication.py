"""Focused tests for executable CONN API authentication helpers."""

import pytest
from fastapi import HTTPException

from capabilities.common.conn.api import get_websocket_user, validate_api_credentials
from capabilities.common.conn.security import AuthenticationManager


class FakeWebSocket:
	def __init__(self, headers=None, query_params=None):
		self.headers = headers or {}
		self.query_params = query_params or {}


def _auth_manager_with_user():
	manager = AuthenticationManager()
	user = manager.create_user(
		username="api_user",
		email="api@example.com",
		password="Str0ng!ApiSignal2026",
		tenant_id="tenant-api",
		roles=["operator"]
	)
	return manager, user


def test_validate_api_credentials_accepts_jwt_bearer_token():
	manager, user = _auth_manager_with_user()
	token = manager.generate_jwt_token(user)

	current_user = validate_api_credentials(f"Bearer {token}", authentication_manager=manager)

	assert current_user["user_id"] == user.user_id
	assert current_user["tenant_id"] == "tenant-api"
	assert current_user["roles"] == ["operator"]
	assert current_user["auth_source"] == "jwt"


def test_validate_api_credentials_accepts_api_key():
	manager, user = _auth_manager_with_user()
	api_key = manager.generate_api_key(user)

	current_user = validate_api_credentials(api_key, authentication_manager=manager)

	assert current_user["user_id"] == user.user_id
	assert current_user["tenant_id"] == "tenant-api"
	assert current_user["session_id"]
	assert current_user["auth_source"] == "api_key"


def test_validate_api_credentials_rejects_invalid_token():
	manager, _user = _auth_manager_with_user()

	with pytest.raises(HTTPException) as exc_info:
		validate_api_credentials("not-a-valid-token", authentication_manager=manager)

	assert exc_info.value.status_code == 401
	assert exc_info.value.detail == "Invalid authentication token"


def test_get_websocket_user_reads_authorization_header():
	manager, user = _auth_manager_with_user()
	token = manager.generate_jwt_token(user)
	websocket = FakeWebSocket(headers={"authorization": f"Bearer {token}"})

	current_user = get_websocket_user(websocket, authentication_manager=manager)

	assert current_user["user_id"] == user.user_id
	assert current_user["tenant_id"] == "tenant-api"
	assert current_user["auth_source"] == "jwt"


def test_get_websocket_user_reads_query_token():
	manager, user = _auth_manager_with_user()
	token = manager.generate_jwt_token(user)
	websocket = FakeWebSocket(query_params={"token": token})

	current_user = get_websocket_user(websocket, authentication_manager=manager)

	assert current_user["user_id"] == user.user_id
	assert current_user["tenant_id"] == "tenant-api"

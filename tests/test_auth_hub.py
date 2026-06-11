"""Tests for APG Auth Hub — interchangeable auth/authz provider adapter.

All tests use the NullProvider which requires no external services.
Provider-specific integration tests (Keycloak, SpiceDB, Clerk, BetterAuth)
are in tests/integration/ and require running services.
"""
import asyncio
import os

import pytest


@pytest.fixture(autouse=True)
def use_null_providers(monkeypatch):
	"""Force null providers for all tests."""
	monkeypatch.setenv("APG_AUTH_PROVIDER", "null")
	monkeypatch.setenv("APG_AUTHZ_PROVIDER", "null")
	# Reset singletons between tests
	from capabilities.common.auth_hub.factory import reset_providers
	reset_providers(_testing_only=True)
	yield
	reset_providers(_testing_only=True)


# ── Factory tests ─────────────────────────────────────────────────

class TestFactory:
	def test_null_provider_created(self):
		from capabilities.common.auth_hub.factory import get_auth_provider, get_authz_provider
		auth = get_auth_provider()
		authz = get_authz_provider()
		assert auth.provider_name == "null"
		assert authz.provider_name == "null"

	def test_provider_singleton(self):
		from capabilities.common.auth_hub.factory import get_auth_provider
		a = get_auth_provider()
		b = get_auth_provider()
		assert a is b  # singleton

	def test_provider_info(self):
		from capabilities.common.auth_hub.factory import provider_info
		info = provider_info()
		assert info["auth_provider"] == "null"
		assert info["authz_provider"] == "null"

	def test_unknown_provider_raises(self, monkeypatch):
		monkeypatch.setenv("APG_AUTH_PROVIDER", "invalid_provider")
		from capabilities.common.auth_hub.factory import reset_providers, _create_auth_provider
		reset_providers(_testing_only=True)
		with pytest.raises(ValueError, match="Unknown APG_AUTH_PROVIDER"):
			_create_auth_provider()


# ── Authentication tests ──────────────────────────────────────────

class TestAuthentication:
	async def test_authenticate_returns_user_and_tokens(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		result = await svc.authenticate({"username": "alice", "password": "pw"})
		assert result.user is not None
		assert result.user.email == "dev@localhost"
		assert result.tokens.access_token
		assert result.tokens.expires_in > 0

	async def test_validate_token_returns_payload(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		result = await svc.authenticate({"username": "alice", "password": "pw"})
		payload = await svc.validate_token(result.tokens.access_token)
		assert payload.user_id == "dev-user"
		assert "admin" in payload.roles

	async def test_empty_token_raises_guard(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		with pytest.raises(ValueError):
			await svc.validate_token("")

	async def test_refresh_token(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		result = await svc.authenticate({"username": "alice", "password": "pw"})
		new_tokens = await svc.refresh_token(result.tokens.refresh_token)
		assert new_tokens.access_token

	async def test_logout(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		result = await svc.authenticate({})
		await svc.logout(result.tokens.access_token)  # should not raise

	async def test_magic_link_null_provider(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		await svc.send_magic_link("user@test.com", "https://app.com/callback")  # no-op
		magic_result = await svc.verify_magic_link("any-token")
		assert magic_result.user is not None

	async def test_mfa_setup_null_provider(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		setup = await svc.setup_mfa("user-123", "totp")
		assert setup.mfa_type == "totp"
		assert setup.secret == "NULLSECRET"


# ── User Management tests ─────────────────────────────────────────

class TestUserManagement:
	async def test_create_and_get_user(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		user = await svc.create_user({"email": "jane@test.com", "roles": ["user"]})
		assert user.email == "jane@test.com"
		assert "user" in user.roles

		fetched = await svc.get_user(user.id)
		assert fetched.id == user.id

	async def test_create_user_requires_email_or_username(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		with pytest.raises(ValueError, match="email or username required"):
			await svc.create_user({})

	async def test_update_user(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		user = await svc.create_user({"email": "update@test.com"})
		updated = await svc.update_user(user.id, {"first_name": "Updated"})
		assert updated.first_name == "Updated"

	async def test_delete_user(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		user = await svc.create_user({"email": "delete@test.com"})
		await svc.delete_user(user.id)
		with pytest.raises(KeyError):
			await svc.get_user(user.id)

	async def test_list_users(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		await svc.create_user({"email": "list1@test.com"})
		await svc.create_user({"email": "list2@test.com"})
		result = await svc.list_users()
		assert result.total >= 2

	async def test_list_users_search(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		await svc.create_user({"email": "search_unique@test.com"})
		result = await svc.list_users(search="search_unique")
		assert any(u.email == "search_unique@test.com" for u in result.users)

	async def test_get_user_empty_id_raises(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		with pytest.raises(ValueError):
			await svc.get_user("")


# ── Authorization tests ───────────────────────────────────────────

class TestAuthorization:
	async def test_check_permission_returns_true_for_null(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService(tenant_id="test_tenant")
		assert await svc.check_permission("user-1", "read")
		assert await svc.check_permission("user-1", "write")
		assert await svc.check_permission("user-1", "admin:delete")

	async def test_check_resource_access(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		assert await svc.check_resource_access("user-1", "document", "doc-123", "edit")

	async def test_assign_and_get_roles(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService(tenant_id="tenant_a")
		await svc.assign_role("user-1", "editor", tenant_id="tenant_a")
		await svc.assign_role("user-1", "viewer", tenant_id="tenant_a")
		roles = await svc.get_user_roles("user-1", tenant_id="tenant_a")
		assert "editor" in roles
		assert "viewer" in roles

	async def test_revoke_role(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService(tenant_id="tenant_b")
		await svc.assign_role("user-2", "editor", tenant_id="tenant_b")
		await svc.revoke_role("user-2", "editor", tenant_id="tenant_b")
		roles = await svc.get_user_roles("user-2", tenant_id="tenant_b")
		assert "editor" not in roles

	async def test_create_and_list_roles(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		role = await svc.create_role("analyst", ["read", "export"])
		assert role["role"] == "analyst"
		roles = await svc.list_roles()
		assert any(r.get("role") == "analyst" for r in roles)

	async def test_bulk_check_permissions(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		results = await svc.bulk_check_permissions(
			"user-1",
			[{"permission": "read"}, {"permission": "write"}, {"permission": "admin"}],
		)
		assert results["read"] is True
		assert results["write"] is True
		assert results["admin"] is True

	async def test_write_and_delete_relationship(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		# No-op for null provider — just verify it doesn't raise
		await svc.write_relationship("document", "doc-1", "owner", "user", "alice")
		await svc.delete_relationship("document", "doc-1", "owner", "user", "alice")

	async def test_list_accessible_resources(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		resources = await svc.list_accessible_resources("user-1", "document", "read")
		assert isinstance(resources, list)

	async def test_empty_user_id_raises_guard(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		with pytest.raises(ValueError):
			await svc.check_permission("", "read")

	async def test_empty_permission_raises_guard(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		with pytest.raises(ValueError):
			await svc.check_permission("user-1", "")


# ── Protocol compliance tests ─────────────────────────────────────

class TestProtocolCompliance:
	"""Verify null providers satisfy the Protocol interface."""

	async def test_null_auth_satisfies_protocol(self):
		from capabilities.common.auth_hub.protocols import AuthProvider
		from capabilities.common.auth_hub.providers.null_provider import NullAuthProvider
		provider = NullAuthProvider()
		assert isinstance(provider, AuthProvider)

	async def test_null_authz_satisfies_protocol(self):
		from capabilities.common.auth_hub.protocols import AuthzProvider
		from capabilities.common.auth_hub.providers.null_provider import NullAuthzProvider
		provider = NullAuthzProvider()
		assert isinstance(provider, AuthzProvider)

	async def test_null_provider_has_provider_name(self):
		from capabilities.common.auth_hub.providers.null_provider import NullAuthProvider, NullAuthzProvider
		assert NullAuthProvider().provider_name == "null"
		assert NullAuthzProvider().provider_name == "null"

	async def test_auth_result_fields(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		result = await svc.authenticate({})
		assert hasattr(result, "user")
		assert hasattr(result, "tokens")
		assert hasattr(result, "mfa_required")
		assert hasattr(result.tokens, "access_token")
		assert hasattr(result.tokens, "refresh_token")
		assert hasattr(result.tokens, "expires_in")

	async def test_token_payload_fields(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		result = await svc.authenticate({})
		payload = await svc.validate_token(result.tokens.access_token)
		assert hasattr(payload, "user_id")
		assert hasattr(payload, "email")
		assert hasattr(payload, "roles")
		assert hasattr(payload, "tenant_id")
		assert hasattr(payload, "is_expired")


# ── Health tests ──────────────────────────────────────────────────

class TestHealth:
	async def test_health_check_returns_ok(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		h = await svc.health_check()
		assert h["status"] == "ok"
		assert "auth_provider" in h
		assert "authz_provider" in h
		assert h["config"]["auth"] == "null"

	async def test_describe(self):
		from capabilities.common.auth_hub import AuthHubService
		svc = AuthHubService()
		desc = await svc.describe()
		assert desc["id"] == "auth_hub"
		assert desc["auth_provider"] == "null"

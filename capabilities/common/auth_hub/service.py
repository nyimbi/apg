"""Auth hub service — unified authentication and authorization facade.

Wraps auth and authz providers and provides a single entry point for all
auth operations. Provider selection is handled by the factory.

Usage:
    from capabilities.common.auth_hub.service import AuthHubService

    svc = AuthHubService()  # uses factory to get configured providers
    result = await svc.authenticate({"username": "alice", "password": "s3cr3t"})
    user = await svc.get_user(result.tokens.access_token)
    allowed = await svc.check_permission(user.id, "payments:write", tenant_id="acme")
"""
from __future__ import annotations

import logging
from typing import Any

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
from .protocols import (
    AuthResult, AuthUser, AuthenticationError, AuthorizationError,
    MFASetup, TokenPair, TokenPayload, UserList,
)

_log = logging.getLogger(__name__)


class AuthHubService:
    """Unified auth/authz service that delegates to the configured providers.

    Instantiate with explicit providers (for testing) or rely on factory
    auto-detection from environment variables.
    """

    def __init__(
        self,
        auth_provider: Any | None = None,
        authz_provider: Any | None = None,
        tenant_id: str = "default",
    ) -> None:
        from .factory import get_auth_provider, get_authz_provider
        self._auth = auth_provider or get_auth_provider()
        self._authz = authz_provider or get_authz_provider()
        self._tenant_id = tenant_id
        self._health_cache = BoundedCache(max_size=10)

    # ── Authentication ────────────────────────────────────────────────

    async def authenticate(self, credentials: dict[str, Any]) -> AuthResult:
        """Verify credentials and return tokens + user.

        Accepts: {username/email, password} | {token} | {code, redirect_uri}
        """
        return await self._auth.authenticate(credentials)

    async def validate_token(self, token: str) -> TokenPayload:
        """Validate an access token. Raises AuthenticationError if invalid."""
        guard_non_empty_string(token, "token")
        return await self._auth.validate_token(token)

    async def refresh_token(self, refresh_token: str) -> TokenPair:
        """Exchange refresh token for new access/refresh pair."""
        guard_non_empty_string(refresh_token, "refresh_token")
        return await self._auth.refresh_token(refresh_token)

    async def logout(self, token: str, refresh_token: str | None = None) -> None:
        """Invalidate the session."""
        guard_non_empty_string(token, "token")
        await self._auth.logout(token, refresh_token)

    # ── User Management ───────────────────────────────────────────────

    async def create_user(self, user_data: dict[str, Any]) -> AuthUser:
        if not user_data.get("email") and not user_data.get("username"):
            raise ValueError("email or username required to create user")
        return await self._auth.create_user(user_data)

    async def get_user(self, user_id: str) -> AuthUser:
        guard_non_empty_string(user_id, "user_id")
        return await self._auth.get_user(user_id)

    async def update_user(self, user_id: str, updates: dict[str, Any]) -> AuthUser:
        guard_non_empty_string(user_id, "user_id")
        return await self._auth.update_user(user_id, updates)

    async def delete_user(self, user_id: str) -> None:
        guard_non_empty_string(user_id, "user_id")
        await self._auth.delete_user(user_id)

    async def list_users(self, search: str | None = None, limit: int = 50, page: int = 1) -> UserList:
        return await self._auth.list_users(search=search, limit=limit, page=page)

    # ── Password / Magic Link ─────────────────────────────────────────

    async def send_password_reset(self, email: str) -> None:
        guard_non_empty_string(email, "email")
        await self._auth.send_password_reset(email)

    async def reset_password(self, token: str, new_password: str) -> None:
        guard_non_empty_string(token, "token")
        guard_non_empty_string(new_password, "new_password")
        await self._auth.reset_password(token, new_password)

    async def send_magic_link(self, email: str, redirect_url: str) -> None:
        guard_non_empty_string(email, "email")
        guard_non_empty_string(redirect_url, "redirect_url")
        await self._auth.send_magic_link(email, redirect_url)

    async def verify_magic_link(self, token: str) -> AuthResult:
        guard_non_empty_string(token, "token")
        return await self._auth.verify_magic_link(token)

    # ── OAuth ─────────────────────────────────────────────────────────

    async def get_oauth_url(
        self, provider: str, redirect_uri: str, state: str, scopes: list[str] | None = None
    ) -> str:
        return await self._auth.get_oauth_authorization_url(provider, redirect_uri, state, scopes)

    async def exchange_oauth_code(
        self, code: str, state: str, redirect_uri: str, provider: str
    ) -> AuthResult:
        return await self._auth.exchange_oauth_code(code, state, redirect_uri, provider)

    # ── MFA ────────────────────────────────────────────────────────────

    async def setup_mfa(self, user_id: str, mfa_type: str = "totp") -> MFASetup:
        guard_non_empty_string(user_id, "user_id")
        return await self._auth.setup_mfa(user_id, mfa_type)

    async def verify_mfa(self, user_id: str, code: str, session_token: str) -> AuthResult:
        return await self._auth.verify_mfa(user_id, code, session_token)

    async def disable_mfa(self, user_id: str, mfa_type: str = "totp") -> None:
        await self._auth.disable_mfa(user_id, mfa_type)

    # ── Sessions ───────────────────────────────────────────────────────

    async def get_sessions(self, user_id: str) -> list[dict[str, Any]]:
        guard_non_empty_string(user_id, "user_id")
        return await self._auth.get_sessions(user_id)

    async def revoke_session(self, session_id: str) -> None:
        guard_non_empty_string(session_id, "session_id")
        await self._auth.revoke_session(session_id)

    # ── Authorization ─────────────────────────────────────────────────

    async def check_permission(
        self,
        user_id: str,
        permission: str,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        resource_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        guard_non_empty_string(user_id, "user_id")
        guard_non_empty_string(permission, "permission")
        tid = tenant_id or self._tenant_id
        return await self._authz.check_permission(
            user_id=user_id,
            permission=permission,
            tenant_id=tid,
            resource_id=resource_id,
            resource_type=resource_type,
            context=context,
        )

    async def check_resource_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        tenant_id: str | None = None,
    ) -> bool:
        guard_non_empty_string(user_id, "user_id")
        tid = tenant_id or self._tenant_id
        return await self._authz.check_resource_access(user_id, resource_type, resource_id, action, tid)

    async def get_user_roles(self, user_id: str, tenant_id: str | None = None) -> list[str]:
        guard_non_empty_string(user_id, "user_id")
        return await self._authz.get_user_roles(user_id, tenant_id or self._tenant_id)

    async def assign_role(
        self, user_id: str, role: str, tenant_id: str | None = None, granted_by: str = "system"
    ) -> None:
        guard_non_empty_string(user_id, "user_id")
        guard_non_empty_string(role, "role")
        await self._authz.assign_role(user_id, role, tenant_id or self._tenant_id, granted_by)

    async def revoke_role(
        self, user_id: str, role: str, tenant_id: str | None = None, revoked_by: str = "system"
    ) -> None:
        await self._authz.revoke_role(user_id, role, tenant_id or self._tenant_id, revoked_by)

    async def create_role(
        self, role: str, permissions: list[str], tenant_id: str | None = None, description: str = ""
    ) -> dict[str, Any]:
        guard_non_empty_string(role, "role")
        return await self._authz.create_role(role, permissions, tenant_id or self._tenant_id, description)

    async def delete_role(self, role: str, tenant_id: str | None = None) -> None:
        await self._authz.delete_role(role, tenant_id or self._tenant_id)

    async def list_roles(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        return await self._authz.list_roles(tenant_id or self._tenant_id)

    async def get_role_permissions(self, role: str, tenant_id: str | None = None) -> list[str]:
        return await self._authz.get_role_permissions(role, tenant_id or self._tenant_id)

    async def write_relationship(
        self, resource_type: str, resource_id: str, relation: str, subject_type: str, subject_id: str
    ) -> None:
        """Write a SpiceDB-style relationship tuple. No-op on RBAC providers."""
        await self._authz.write_relationship(resource_type, resource_id, relation, subject_type, subject_id)

    async def delete_relationship(
        self, resource_type: str, resource_id: str, relation: str, subject_type: str, subject_id: str
    ) -> None:
        await self._authz.delete_relationship(resource_type, resource_id, relation, subject_type, subject_id)

    async def list_accessible_resources(
        self, user_id: str, resource_type: str, action: str, tenant_id: str | None = None
    ) -> list[str]:
        return await self._authz.list_accessible_resources(
            user_id, resource_type, action, tenant_id or self._tenant_id
        )

    async def bulk_check_permissions(
        self, user_id: str, checks: list[dict[str, Any]], tenant_id: str | None = None
    ) -> dict[str, bool]:
        return await self._authz.bulk_check_permissions(
            user_id, checks, tenant_id or self._tenant_id
        )

    # ── Health ────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        auth_health = await self._auth.health_check()
        authz_health = await self._authz.health_check()
        overall = "ok" if auth_health.get("status") == "ok" and authz_health.get("status") == "ok" else "degraded"
        return {
            "status": overall,
            "auth_provider": auth_health,
            "authz_provider": authz_health,
            "config": {
                "auth": self._auth.provider_name,
                "authz": self._authz.provider_name,
            },
        }

    async def describe(self) -> dict[str, Any]:
        return {
            "id": "auth_hub",
            "name": "Authentication Hub",
            "domain": "common",
            "version": "1.0.0",
            "auth_provider": self._auth.provider_name,
            "authz_provider": self._authz.provider_name,
        }
